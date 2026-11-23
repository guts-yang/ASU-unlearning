"""Optimization loop: Idea1 loss variants x lambda, after stage8 primary gate failed."""
import argparse, copy, json, random, statistics
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from counterfactual import build_pairs, load_jsonl
from stage1_minimal import encode
from stage3_sweep import evaluate

def train_one(args, tok, pairs_train, retain_train, P, mode, lam, seed, device):
    rng = random.Random(seed); torch.manual_seed(seed)
    tp, tr = pairs_train[:], retain_train[:]
    rng.shuffle(tp); rng.shuffle(tr)
    student = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    teacher = copy.deepcopy(student).eval().to(device)
    for p in teacher.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    for step in range(args.steps):
        pair = tp[step % len(tp)]; rp = tr[step % len(tr)]
        ids_plus = encode(tok, pair['x_plus'], device)
        q_ids = encode(tok, rp['question'], device); a_ids = encode(tok, rp['answer'], device)
        with torch.no_grad():
            t_h = teacher(input_ids=ids_plus, output_hidden_states=True).hidden_states[args.layer][0].float().detach()
        s_h = student(input_ids=ids_plus, output_hidden_states=True).hidden_states[args.layer][0].float()
        if mode == 'suppress':
            proj = s_h @ P
            lf = (proj*proj).sum(-1).mean()
        else:
            diff = s_h - t_h
            proj = diff @ P
            lf = (diff - proj @ P.T).pow(2).sum(-1).mean()
        r_out = student(input_ids=torch.cat([q_ids, a_ids], dim=1))
        r_logits = r_out.logits[:, q_ids.size(1)-1:-1, :]
        l_r = F.cross_entropy(r_logits.reshape(-1, r_logits.size(-1)), a_ids.reshape(-1))
        loss = args.beta * l_r + lam * lf
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
    return student

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full')
    ap.add_argument('--forget', default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json')
    ap.add_argument('--retain', default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json')
    ap.add_argument('--basis', default='da_basis_pooled_layer16.npy')
    ap.add_argument('--pairs', type=int, default=36); ap.add_argument('--train-pairs', type=int, default=18)
    ap.add_argument('--steps', type=int, default=40); ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--lr', type=float, default=1e-5); ap.add_argument('--beta', type=float, default=8.0)
    ap.add_argument('--layer', type=int, default=16)
    ap.add_argument('--modes', default='suppress,orthalign')
    ap.add_argument('--lams', default='0.1,0.2,0.4,0.7,1.0')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', default='stage9_idea1_opt_result.json')
    args = ap.parse_args()
    device = torch.device(args.device); seeds = [int(s) for s in args.seeds.split(',')]
    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    retain = load_jsonl(args.retain)
    P = torch.tensor(np.load(args.basis), dtype=torch.float32, device=device)
    tp, fp = pairs[:args.train_pairs], pairs[args.train_pairs:]
    tr, fr = retain[:100], retain[100:108]
    results = {}
    ref_f, ref_r = 0.5551, 0.7985
    i4_f, i4_r = 0.6507, 1.0117
    target_r = ref_r + 0.5*(i4_r-ref_r)
    for mode in args.modes.split(','):
        for lam in [float(x) for x in args.lams.split(',')]:
            runs = []
            for seed in seeds:
                m = train_one(args, tok, tp, tr, P, mode, lam, seed, device)
                f, r = evaluate(m, tok, fp, fr, device)
                runs.append({'seed': seed, 'forget': f, 'retain': r})
                del m; torch.cuda.empty_cache()
            fm = statistics.mean(x['forget'] for x in runs); rm = statistics.mean(x['retain'] for x in runs)
            ok = (fm >= i4_f) and (rm <= target_r)
            key = f'{mode}_lam{lam}'
            results[key] = {'forget': fm, 'retain': rm, 'forget_std': statistics.pstdev(x['forget'] for x in runs),
                            'retain_std': statistics.pstdev(x['retain'] for x in runs), 'pass_gate': ok, 'runs': runs}
            print(f'{key}: forget {fm:.4f} retain {rm:.4f} gate={"PASS" if ok else "FAIL"}', flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print('wrote', args.output, flush=True)

if __name__ == '__main__':
    main()
