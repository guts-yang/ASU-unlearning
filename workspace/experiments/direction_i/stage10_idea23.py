"""GPU reproduction of Idea2 (difference matching) and Idea3 (prediction dynamics)."""
import argparse, copy, json, random, statistics
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from counterfactual import build_pairs, load_jsonl
from stage1_minimal import encode
from stage3_sweep import evaluate

def train_one(args, tok, pairs_train, retain_train, arm, seed, device):
    rng = random.Random(seed); torch.manual_seed(seed)
    tp, tr = pairs_train[:], retain_train[:]
    rng.shuffle(tp); rng.shuffle(tr)
    student = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    teacher = copy.deepcopy(student).eval().to(device)
    for p in teacher.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    for step in range(args.steps):
        pair = tp[step % len(tp)]; rp = tr[step % len(tr)]
        xp = encode(tok, pair['x_plus'], device); xm = encode(tok, pair['x_minus'], device)
        q = encode(tok, rp['question'], device); a = encode(tok, rp['answer'], device)
        loss = args.beta * F.cross_entropy(
            student(input_ids=torch.cat([q,a],dim=1)).logits[:,q.size(1)-1:-1,:].reshape(-1, student.config.vocab_size),
            a.reshape(-1))
        if arm == 'idea2':
            with torch.no_grad():
                hp = teacher(input_ids=xp, output_hidden_states=True).hidden_states[args.layer][0].float().detach()
                hm = teacher(input_ids=xm, output_hidden_states=True).hidden_states[args.layer][0].float().detach()
            sp = student(input_ids=xp, output_hidden_states=True).hidden_states[args.layer][0].float()
            sm = student(input_ids=xm, output_hidden_states=True).hidden_states[args.layer][0].float()
            L = min(len(hp),len(hm),len(sp),len(sm))
            dT = (hp[:L]-hm[:L]); dS = (sp[:L]-sm[:L])
            lf = F.mse_loss(dS.mean(0), dT.mean(0))
            loss = loss + args.lam * lf
        elif arm == 'idea3':
            with torch.no_grad():
                th = [teacher(input_ids=xm, output_hidden_states=True).hidden_states[l][0].float().detach() for l in args.dyn_layers]
            sh = [student(input_ids=xp, output_hidden_states=True).hidden_states[l][0].float() for l in args.dyn_layers]
            cos = []
            for i in range(len(th)-1):
                dt = th[i+1]-th[i]; ds = sh[i+1]-sh[i]
                dt = dt.mean(0); ds = ds.mean(0)
                cos.append(F.cosine_similarity(ds.unsqueeze(0), dt.unsqueeze(0)).squeeze())
            lf = 1.0 - torch.stack(cos).mean()
            loss = loss + args.lam * lf
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
    return student

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full')
    ap.add_argument('--forget', default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json')
    ap.add_argument('--retain', default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json')
    ap.add_argument('--pairs', type=int, default=36); ap.add_argument('--train-pairs', type=int, default=18)
    ap.add_argument('--steps', type=int, default=40); ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--lr', type=float, default=1e-5); ap.add_argument('--beta', type=float, default=8.0)
    ap.add_argument('--lam', type=float, default=1.0); ap.add_argument('--layer', type=int, default=16)
    ap.add_argument('--dyn-layers', default='10,12,14,16'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', default='stage10_idea23_result.json')
    args = ap.parse_args()
    args.dyn_layers = [int(x) for x in args.dyn_layers.split(',')]
    device = torch.device(args.device); seeds=[int(s) for s in args.seeds.split(',')]
    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = build_pairs(tok,args.forget,args.retain,max_pairs=args.pairs,seed=0)
    retain = load_jsonl(args.retain)
    tp,fp = pairs[:args.train_pairs],pairs[args.train_pairs:]; tr,fr = retain[:100],retain[100:108]
    results={}
    for arm in ('ref','idea2','idea3'):
        runs=[]
        for seed in seeds:
            m=train_one(args,tok,tp,tr,arm,seed,device)
            f,r=evaluate(m,tok,fp,fr,device)
            runs.append({'seed':seed,'forget':f,'retain':r})
            del m; torch.cuda.empty_cache()
        fm=statistics.mean(x['forget'] for x in runs); rm=statistics.mean(x['retain'] for x in runs)
        results[arm]={'forget':fm,'retain':rm,'forget_std':statistics.pstdev(x['forget'] for x in runs),'retain_std':statistics.pstdev(x['retain'] for x in runs),'runs':runs}
        print(f"{arm}: forget {fm:.4f} retain {rm:.4f}",flush=True)
    Path(args.output).write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print('wrote',args.output,flush=True)
if __name__=='__main__': main()
