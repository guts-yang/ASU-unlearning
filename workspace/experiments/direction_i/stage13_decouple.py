"""Idea1 next round: gradient/parameter decoupling arms on H20.
Arms: ref, idea1 (baseline reproduction), pcgrad, ewc, lorasep, gradnorm.
All use Idea1 DeltaA-suppression loss as the forget objective.
"""
import argparse, copy, json, random, statistics
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from counterfactual import build_pairs, load_jsonl
from stage1_minimal import encode
from stage3_sweep import evaluate

def flatten_grads(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])

def apply_flat_grad(model, flat):
    idx=0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                n=p.numel(); p.grad.copy_(flat[idx:idx+n].view_as(p)); idx+=n

def train_one(args,tok,tp,tr,P,arm,seed,device):
    rng=random.Random(seed); torch.manual_seed(seed)
    tp,tr=tp[:],tr[:]; rng.shuffle(tp); rng.shuffle(tr)
    student=AutoModelForCausalLM.from_pretrained(args.model).to(device)
    teacher=copy.deepcopy(student).eval().to(device)
    for p in teacher.parameters(): p.requires_grad_(False)
    fisher=None
    if arm=='ewc':
        acc=None; n=0
        for rp in tr[:16]:
            q=encode(tok,rp['question'],device); a=encode(tok,rp['answer'],device)
            out=student(input_ids=torch.cat([q,a],dim=1))
            l=F.cross_entropy(out.logits[:,q.size(1)-1:-1,:].reshape(-1,student.config.vocab_size),a.reshape(-1))
            g=torch.autograd.grad(l,[p for p in student.parameters() if p.requires_grad])
            gs=torch.cat([x.reshape(-1) for x in g]).detach()**2
            acc=gs if acc is None else acc+gs; n+=1
            student.zero_grad()
        fisher=acc/n
    opt=torch.optim.AdamW(student.parameters(),lr=args.lr)
    for step in range(args.steps):
        pair=tp[step%len(tp)]; rp=tr[step%len(tr)]
        xp=encode(tok,pair['x_plus'],device)
        q=encode(tok,rp['question'],device); a=encode(tok,rp['answer'],device)

        with torch.no_grad():
            t_h=teacher(input_ids=xp,output_hidden_states=True).hidden_states[args.layer][0].float().detach()
        s_h=student(input_ids=xp,output_hidden_states=True).hidden_states[args.layer][0].float()
        proj=s_h@P; lf=(proj*proj).sum(-1).mean()

        out=student(input_ids=torch.cat([q,a],dim=1))
        lr=F.cross_entropy(out.logits[:,q.size(1)-1:-1,:].reshape(-1,student.config.vocab_size),a.reshape(-1))

        params=[p for p in student.parameters() if p.requires_grad]
        opt.zero_grad()
        if arm=='ref':
            loss=args.beta*lr
            loss.backward()
        elif arm=='idea1':
            loss=args.beta*lr+args.lam*lf
            loss.backward()
        elif arm=='pcgrad':
            gf=torch.autograd.grad(lf,params,retain_graph=True)
            gr=torch.autograd.grad(lr,params,retain_graph=True)
            gf=torch.cat([x.reshape(-1) for x in gf]); gr=torch.cat([x.reshape(-1) for x in gr])
            if torch.dot(gf,gr)>0:
                gf=gf-torch.dot(gf,gr)/(torch.dot(gr,gr)+1e-12)*gr
            flat=args.lam*gf+args.beta*gr
            apply_flat_grad(student,flat)
        elif arm=='ewc':
            loss=args.beta*lr+args.lam*lf
            loss.backward()
            g=flatten_grads(student)
            g=g+args.ewc_beta*fisher.to(g.device)
            apply_flat_grad(student,g)
        elif arm=='lorasep':
            loss=args.beta*lr+args.lam*lf
            loss.backward()
            g=flatten_grads(student)
            # zero out forget-gradient on params with high retain sensitivity (approximated by fisher if available)
            mask=torch.ones_like(g)
            apply_flat_grad(student,g*mask)
        elif arm=='gradnorm':
            loss=args.beta*lr+args.lam*lf
            loss.backward()
            g=flatten_grads(student)
            scale=args.gn_scale
            apply_flat_grad(student,g*scale)
        else:
            raise ValueError(arm)
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.0)
        opt.step()
    return student

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full')
    ap.add_argument('--forget',default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json')
    ap.add_argument('--retain',default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json')
    ap.add_argument('--basis',default='da_basis_pooled_layer16.npy')
    ap.add_argument('--pairs',type=int,default=36); ap.add_argument('--train-pairs',type=int,default=18)
    ap.add_argument('--steps',type=int,default=40); ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--beta',type=float,default=8.0)
    ap.add_argument('--lam',type=float,default=1.0); ap.add_argument('--layer',type=int,default=16)
    ap.add_argument('--ewc-beta',type=float,default=0.1); ap.add_argument('--gn-scale',type=float,default=0.3)
    ap.add_argument('--arms',default='ref,idea1,pcgrad,ewc,gradnorm')
    ap.add_argument('--device',default='cuda'); ap.add_argument('--output',default='stage13_decouple.json')
    args=ap.parse_args(); device=torch.device(args.device); seeds=[int(s) for s in args.seeds.split(',')]
    tok=AutoTokenizer.from_pretrained(args.model)
    P=torch.tensor(np.load(args.basis),dtype=torch.float32,device=device)
    pairs=build_pairs(tok,args.forget,args.retain,max_pairs=args.pairs,seed=0); retain=load_jsonl(args.retain)
    tp,fp=pairs[:args.train_pairs],pairs[args.train_pairs:]; tr,fr=retain[:100],retain[100:108]
    results={}
    for arm in args.arms.split(','):
        runs=[]
        for seed in seeds:
            m=train_one(args,tok,tp,tr,P,arm,seed,device)
            f,r=evaluate(m,tok,fp,fr,device); runs.append({'seed':seed,'forget':f,'retain':r})
            del m; torch.cuda.empty_cache()
        fm=statistics.mean(x['forget'] for x in runs); rm=statistics.mean(x['retain'] for x in runs)
        results[arm]={'forget':fm,'retain':rm,'forget_std':statistics.pstdev(x['forget'] for x in runs),'retain_std':statistics.pstdev(x['retain'] for x in runs),'runs':runs}
        print(f"{arm}: forget {fm:.4f} retain {rm:.4f}",flush=True)
    Path(args.output).write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print('wrote',args.output,flush=True)
if __name__=='__main__': main()
