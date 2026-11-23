"""New-A (remote-agent fallback): logits-level counterfactual response methods."""
import argparse, copy, json, random, statistics
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from counterfactual import build_pairs, load_jsonl
from stage1_minimal import encode
from stage3_sweep import evaluate

def train_one(args, tok, pairs_train, retain_train, arm, seed, device):
    rng=random.Random(seed); torch.manual_seed(seed)
    tp,tr=pairs_train[:],retain_train[:]; rng.shuffle(tp); rng.shuffle(tr)
    student=AutoModelForCausalLM.from_pretrained(args.model).to(device)
    teacher=copy.deepcopy(student).eval().to(device)
    for p in teacher.parameters(): p.requires_grad_(False)
    opt=torch.optim.AdamW(student.parameters(),lr=args.lr)
    for step in range(args.steps):
        pair=tp[step%len(tp)]; rp=tr[step%len(tr)]
        xp=encode(tok,pair['x_plus'],device); xm=encode(tok,pair['x_minus'],device)
        ans=encode(tok,pair['answer_plus'],device)
        q=encode(tok,rp['question'],device); a=encode(tok,rp['answer'],device)
        lr=F.cross_entropy(student(input_ids=torch.cat([q,a],dim=1)).logits[:,q.size(1)-1:-1,:].reshape(-1,student.config.vocab_size),a.reshape(-1))
        loss=args.beta*lr
        if arm=='out_idea4':
            with torch.no_grad():
                tl=teacher(input_ids=xm).logits[:,-1,:].float().detach()/args.temp
            sl=student(input_ids=xp).logits[:,-1,:].float()/args.temp
            lf=F.kl_div(F.log_softmax(sl,dim=-1),F.softmax(tl,dim=-1),reduction='batchmean')
            loss=loss+args.lam*lf
        elif arm=='ga_gd':
            full=torch.cat([xp,ans],dim=1)
            forget_ce=F.cross_entropy(student(input_ids=full).logits[:,xp.size(1)-1:-1,:].reshape(-1,student.config.vocab_size),ans.reshape(-1))
            loss=loss-args.lam*forget_ce
        elif arm=='collapse':
            lp=student(input_ids=xp).logits[:,-1,:].float(); lm=student(input_ids=xm).logits[:,-1,:].float()
            lf=1.0-F.cosine_similarity(lp,lm).mean()
            loss=loss+args.lam*lf
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.0); opt.step()
    return student

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full')
    ap.add_argument('--forget',default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json')
    ap.add_argument('--retain',default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json')
    ap.add_argument('--pairs',type=int,default=36); ap.add_argument('--train-pairs',type=int,default=18)
    ap.add_argument('--steps',type=int,default=40); ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--beta',type=float,default=8.0)
    ap.add_argument('--lam',type=float,default=1.0); ap.add_argument('--temp',type=float,default=2.0)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--output',default='stage11_newa_result.json')
    args=ap.parse_args(); device=torch.device(args.device); seeds=[int(s) for s in args.seeds.split(',')]
    tok=AutoTokenizer.from_pretrained(args.model)
    pairs=build_pairs(tok,args.forget,args.retain,max_pairs=args.pairs,seed=0); retain=load_jsonl(args.retain)
    tp,fp=pairs[:args.train_pairs],pairs[args.train_pairs:]; tr,fr=retain[:100],retain[100:108]
    results={}
    for arm in ('ref','out_idea4','ga_gd','collapse'):
        runs=[]
        for seed in seeds:
            m=train_one(args,tok,tp,tr,arm,seed,device)
            f,r=evaluate(m,tok,fp,fr,device); runs.append({'seed':seed,'forget':f,'retain':r})
            del m; torch.cuda.empty_cache()
        fm=statistics.mean(x['forget'] for x in runs); rm=statistics.mean(x['retain'] for x in runs)
        results[arm]={'forget':fm,'retain':rm,'forget_std':statistics.pstdev(x['forget'] for x in runs),'retain_std':statistics.pstdev(x['retain'] for x in runs),'runs':runs}
        print(f"{arm}: forget {fm:.4f} retain {rm:.4f}",flush=True)
    Path(args.output).write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print('wrote',args.output,flush=True)
if __name__=='__main__': main()
