"""Block C (after pooled probe PASS): Idea1 orthogonal-suppression comparison.

Arms: ref, idea4 (direct A-), idea1 (true-donor DeltaA subspace),
      random subspace, wrong-donor subspace.
Fixed: layer16, rank8, beta=8, lam=1, 40 steps, 3 seeds, held-out eval.
"""
import argparse, copy, json, random, re, statistics
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from counterfactual import build_pairs, load_jsonl, token_ids
from stage1_minimal import answer_nll, encode
from stage3_sweep import evaluate
from stage7b_da_probe_pooled import pooled_delta
from stage7_da_probe import pca_basis

NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,2}\b")

def get_bases(args, tok, model, pairs, retain, device):
    d = model.config.hidden_size
    bases = {}
    true = np.load(Path(args.output).with_name('da_basis_pooled_layer16.npy'))
    bases['idea1'] = true
    rng = np.random.default_rng(0)
    bases['random'], _ = np.linalg.qr(rng.standard_normal((d, args.rank)))
    tmpls = {}
    for r in retain:
        tmpls.setdefault(re.sub(NAME_RE, '<NAME>', r['question']), []).append(r)
    keys = list(tmpls)
    wxp, wxm = [], []
    for i, p in enumerate(pairs):
        t = re.sub(NAME_RE, '<NAME>', p['x_plus'])
        other = keys[(keys.index(t) + 11 + i) % len(keys)] if t in keys else keys[i % len(keys)]
        pool = set()
        for r in tmpls[other]:
            for n in NAME_RE.findall(r['question']):
                pool.add(re.sub(r"'s$", '', n))
        pool = [n for n in pool if len(token_ids(tok, n)) == len(token_ids(tok, p['forget_subject']))]
        if not pool: continue
        donor = pool[(i * 5) % len(pool)]
        wxp.append(p['x_plus']); wxm.append(p['x_plus'].replace(p['forget_subject'], donor, 1))
    Dw = pooled_delta(model, tok, wxp, wxm, 16, device)
    bases['wrong'], *_ = pca_basis(Dw, args.rank)
    return bases

def train_one(args, tok, pairs_train, retain_train, arm, basis, seed, device):
    rng = random.Random(seed); torch.manual_seed(seed)
    tp, tr = pairs_train[:], retain_train[:]
    rng.shuffle(tp); rng.shuffle(tr)
    student = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    teacher = copy.deepcopy(student).eval().to(device)
    for p in teacher.parameters(): p.requires_grad_(False)
    P = torch.tensor(basis, dtype=torch.float32, device=device) if basis is not None else None
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    for step in range(args.steps):
        pair = tp[step % len(tp)]; rp = tr[step % len(tr)]
        ids_plus = encode(tok, pair['x_plus'], device); ids_minus = encode(tok, pair['x_minus'], device)
        q_ids = encode(tok, rp['question'], device); a_ids = encode(tok, rp['answer'], device)
        r_out = student(input_ids=torch.cat([q_ids, a_ids], dim=1))
        r_logits = r_out.logits[:, q_ids.size(1)-1:-1, :]
        l_r = F.cross_entropy(r_logits.reshape(-1, r_logits.size(-1)), a_ids.reshape(-1))
        loss = args.beta * l_r
        if arm != 'ref':
            with torch.no_grad():
                out = teacher(input_ids=ids_minus, output_hidden_states=True)
                t_hidden = out.hidden_states[args.layer][0].float().detach()
            out_s = student(input_ids=ids_plus, output_hidden_states=True)
            s_hidden = out_s.hidden_states[args.layer][0].float()
            if arm == 'idea4':
                lf = F.mse_loss(s_hidden[-1], t_hidden[-1])
            else:
                proj = s_hidden @ P
                lf = (proj * proj).sum(-1).mean()
            loss = loss + args.lam * lf
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
    return student

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full')
    ap.add_argument('--forget', default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json')
    ap.add_argument('--retain', default='/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json')
    ap.add_argument('--pairs', type=int, default=36)
    ap.add_argument('--train-pairs', type=int, default=18)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--beta', type=float, default=8.0)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--layer', type=int, default=16)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', default='stage8_idea1_result.json')
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(',')]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    retain = load_jsonl(args.retain)
    bases = get_bases(args, tok, model, pairs, retain, device)
    del model; torch.cuda.empty_cache()
    tp, fp = pairs[:args.train_pairs], pairs[args.train_pairs:]
    tr, fr = retain[:100], retain[100:108]
    base = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    baseline = evaluate(base, tok, fp, fr, device)
    del base; torch.cuda.empty_cache()
    print(f'baseline forget={baseline[0]:.4f} retain={baseline[1]:.4f}', flush=True)
    results = {'baseline': {'forget': baseline[0], 'retain': baseline[1]}, 'arms': {}}
    for arm in ('ref', 'idea4', 'idea1', 'random', 'wrong'):
        runs = []
        for seed in seeds:
            m = train_one(args, tok, tp, tr, arm, bases.get(arm), seed, device)
            f, r = evaluate(m, tok, fp, fr, device)
            runs.append({'seed': seed, 'forget': f, 'retain': r})
            del m; torch.cuda.empty_cache()
        fm = statistics.mean(x['forget'] for x in runs); rm = statistics.mean(x['retain'] for x in runs)
        results['arms'][arm] = {'forget_mean': fm, 'retain_mean': rm,
            'forget_std': statistics.pstdev(x['forget'] for x in runs),
            'retain_std': statistics.pstdev(x['retain'] for x in runs), 'runs': runs}
        print(f"{arm}: forget {fm:.4f} retain {rm:.4f}", flush=True)
    ref = results['arms']['ref']; i4 = results['arms']['idea4']; i1 = results['arms']['idea1']
    results['verdict'] = {
        'idea1_forget_gain_ge_idea4': bool(i1['forget_mean'] >= i4['forget_mean']),
        'idea1_retain_damage_le_half_idea4': bool((i1['retain_mean']-ref['retain_mean']) <= 0.5*(i4['retain_mean']-ref['retain_mean'])),
        'idea1_better_than_random': bool(i1['forget_mean'] > results['arms']['random']['forget_mean']),
    }
    print('VERDICT:', results['verdict'], flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print('wrote', args.output, flush=True)

if __name__ == '__main__':
    main()
