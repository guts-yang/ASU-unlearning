"""Autoregressive optimization: GA+GD lambda sweep after stage11 showed best ratio."""
import argparse, json, statistics
from pathlib import Path
from stage11_newa import train_one
from stage3_sweep import evaluate
from transformers import AutoTokenizer
from counterfactual import build_pairs, load_jsonl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full')
    ap.add_argument('--forget',default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json')
    ap.add_argument('--retain',default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json')
    ap.add_argument('--pairs',type=int,default=36); ap.add_argument('--train-pairs',type=int,default=18)
    ap.add_argument('--steps',type=int,default=40); ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--lr',type=float,default=1e-5); ap.add_argument('--beta',type=float,default=8.0)
    ap.add_argument('--lams',default='0.1,0.3,0.5,1,2,4'); ap.add_argument('--temp',type=float,default=2.0)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--output',default='stage12_ga_sweep.json')
    args=ap.parse_args(); args.lam=1.0
    device=__import__('torch').device(args.device); seeds=[int(s) for s in args.seeds.split(',')]
    tok=AutoTokenizer.from_pretrained(args.model)
    pairs=build_pairs(tok,args.forget,args.retain,max_pairs=args.pairs,seed=0); retain=load_jsonl(args.retain)
    tp,fp=pairs[:args.train_pairs],pairs[args.train_pairs:]; tr,fr=retain[:100],retain[100:108]
    ref_f,ref_r=0.5551,0.7985
    results={}
    for lam in [float(x) for x in args.lams.split(',')]:
        runs=[]
        args.lam=lam
        for seed in seeds:
            m=train_one(args,tok,tp,tr,'ga_gd',seed,device)
            f,r=evaluate(m,tok,fp,fr,device); runs.append({'seed':seed,'forget':f,'retain':r})
            del m; __import__('torch').cuda.empty_cache()
        fm=statistics.mean(x['forget'] for x in runs); rm=statistics.mean(x['retain'] for x in runs)
        ratio=(rm-ref_r)/max(fm-ref_f,1e-6)
        results[str(lam)]={'forget':fm,'retain':rm,'ratio':ratio,
                           'pass_gate':bool(fm-ref_f>=0.05 and ratio<=1.4),'runs':runs}
        print(f"lam={lam}: forget {fm:.4f} retain {rm:.4f} ratio {ratio:.2f}",flush=True)
    Path(args.output).write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print('wrote',args.output,flush=True)
if __name__=='__main__': main()
