"""Fit the router's weights and thresholds on the decision model's cached answers
(tev1's by default; ROUTER_MODEL=laya fits laya's, into weights-laya.json).

Errors are not equal: an opus task sent lower (the work fails or is shallow)
costs 3, an easy task sent to opus (needless spend) costs 2, anything else
costs 1. Fitted on two thirds of the cases and reported on the held-out
third, then refitted on all of them for weights.json.

  python3 -I eval/tune.py            # report only
  python3 -I eval/tune.py --write    # also write tier_router/weights.json
"""
import itertools, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier_router.classify import NEGATIVE, POSITIVE, TIERS, features, score, tier_for, weights_file
from collect import cache_path, key, load_blind, load_cases

RANK = {t: i for i, t in enumerate(TIERS)}


def cost(want, got):
    if want == got:
        return 0
    if want == "opus":
        return 3
    if got == "opus":
        return 2
    return 1


def report(name, rows, pick):
    n = len(rows)
    total = ok = miss = over = 0
    confusion = {w: {g: 0 for g in TIERS} for w in TIERS}
    for r in rows:
        got = pick(r)
        confusion[r["label"]][got] += 1
        total += cost(r["label"], got)
        ok += got == r["label"]
        miss += r["label"] == "opus" and got != "opus"
        over += r["label"] != "opus" and got == "opus"
    n_opus = sum(r["label"] == "opus" for r in rows)
    print(f"{name:<28} acc {ok}/{n} ({ok / n:.0%})  opus missed {miss}/{n_opus}  "
          f"sent to opus needlessly {over}/{n - n_opus}  cost {total}")
    return total, confusion


VALUES = [0, 0.25, 0.5, 1, 2]
NAMES = [f"w_{n}" for n in POSITIVE + NEGATIVE]


def best_cuts(scores):
    """Best (cost, t_sonnet, t_opus) for scored rows, in one pass over them
    sorted: everything below the first cut is haiku, from the second opus."""
    scores = sorted(scores)
    n = len(scores)
    pre = {t: [0] * (n + 1) for t in TIERS}
    for i, (_, lab) in enumerate(scores):
        for t in TIERS:
            pre[t][i + 1] = pre[t][i] + cost(lab, t)
    cut = lambda i: 0.0 if i == 0 else 1.0001 if i == n else (scores[i - 1][0] + scores[i][0]) / 2
    best = None
    best_i, best_i_val = 0, pre["haiku"][0] - pre["sonnet"][0]
    for j in range(n + 1):
        v = pre["haiku"][j] - pre["sonnet"][j]
        if v < best_i_val:
            best_i, best_i_val = j, v
        c = best_i_val + pre["sonnet"][j] + pre["opus"][n] - pre["opus"][j]
        if best is None or c < best[0]:
            best = (c, cut(best_i), cut(j))
    return best


def evaluate(rows, w):
    scores = [(score(r["f"], {**w, "t_sonnet": 0, "t_opus": 1}), r["label"]) for r in rows]
    if sum(w[n] for n in NAMES if n[2:] in ("tier", "deep", "effort")) == 0:
        return (10**9,), None
    c, ts, to = best_cuts(scores)
    size = sum(w[n] > 0 for n in NAMES)  # ties go to the simpler model
    return (c, size), {**w, "t_sonnet": ts, "t_opus": to}


def fit(rows, restarts=12, seed=1):
    """Coordinate descent over VALUES for each weight, from random starts."""
    rnd = random.Random(seed)
    best = None
    for k in range(restarts):
        w = {n: (1 if k == 0 else rnd.choice(VALUES)) for n in NAMES}
        cur = evaluate(rows, w)
        improved = True
        while improved:
            improved = False
            for n in NAMES:
                for v in VALUES:
                    if v == w[n]:
                        continue
                    cand = evaluate(rows, {**w, n: v})
                    if cand[0] < cur[0]:
                        w, cur, improved = {**w, n: v}, cand, True
        if best is None or cur[0] < best[0]:
            best = cur
    return best[1]


def pick_with(w):
    return lambda r: tier_for(score(r["f"], w), w)


def main():
    cache = json.loads(cache_path().read_text())
    rows = []
    for c in load_cases():
        a = cache[key(c)]
        rows.append({**c, "a": a, "f": features(c["prompt"], a)})

    print(f"{len(rows)} cases\n")
    report("model's tier choice only", rows, lambda r: r["a"]["tier"]["choice"])
    report("effort score only", rows,
           lambda r: "haiku" if r["a"]["effort"]["score"] < 1.5 else "sonnet" if r["a"]["effort"]["score"] < 2.6 else "opus")

    # 5-fold cross-validation: every case is judged by weights that never saw it.
    random.seed(7)
    idx = list(range(len(rows)))
    random.shuffle(idx)
    held = {}
    for k in range(5):
        test = set(idx[k::5])
        w = fit([r for i, r in enumerate(rows) if i not in test])
        for i in test:
            held[i] = pick_with(w)(rows[i])
    print()
    cv_rows = [{**r, "_got": held[i]} for i, r in enumerate(rows)]
    report("fitted, 5-fold HELD-OUT", cv_rows, lambda r: r["_got"])

    print("\nrefitting on all …")
    w_all = fit(rows)
    print("weights:", json.dumps(w_all))
    _, conf = report("fitted on all (in-sample)", rows, lambda r: tier_for(score(r["f"], w_all), w_all))
    print("confusion (rows = wanted, cols = got):", json.dumps(conf))
    print("\nmistakes:")
    for r in rows:
        got = tier_for(score(r["f"], w_all), w_all)
        if got != r["label"]:
            print(f"  want {r['label']:<6} got {got:<6} s={score(r['f'], w_all):.2f} [{r['source']}] {r['prompt'][:80]}")
    if "--blind" in sys.argv:
        blind = []
        for c in load_blind():
            a = cache[key(c)]
            blind.append({**c, "a": a, "f": features(c["prompt"], a)})
        print()
        report("BLIND, model's choice only", blind, lambda r: r["a"]["tier"]["choice"])
        report("BLIND, fitted router", blind, pick_with(w_all))
        for r in blind:
            got = pick_with(w_all)(r)
            if got != r["label"]:
                print(f"  want {r['label']:<6} got {got:<6} {r['prompt'][:80]}")
    if "--write" in sys.argv:
        out = weights_file()
        out.write_text(json.dumps(w_all, indent=2) + "\n")
        print(f"\nwrote {out}")
        from tier_router import export_ts
        export_ts.main()


if __name__ == "__main__":
    main()
