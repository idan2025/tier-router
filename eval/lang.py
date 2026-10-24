"""Score the shipped router on the same prompts in two languages.

hebrew_synthetic.tsv and hebrew_blind.tsv are translations of synthetic.tsv and
blind.tsv, line for line, with the same labels. A gap between a pair is the
cost of the language, since the tasks are the same.

  python3 -I eval/lang.py                       # English vs Hebrew, both sets
  python3 -I eval/lang.py eval/some_other.tsv   # any label<TAB>prompt files

tev1's answers are cached in eval/cache.json, like collect.py does.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier_router.classify import ask_tev1, features, load_weights, score, tier_for
from collect import HERE, key
from tune import cost

DEFAULT = ["synthetic.tsv", "hebrew_synthetic.tsv", "blind.tsv", "hebrew_blind.tsv"]


def rows_of(path, cache):
    rows = []
    for line in open(path, encoding="utf-8"):
        label, prompt = line.rstrip("\n").split("\t", 1)
        c = {"prompt": prompt, "context": ""}
        k = key(c)
        if k not in cache:
            cache[k] = ask_tev1(prompt, "", timeout=120)
        rows.append((label, cache[k]))
    return rows


def main():
    paths = [Path(a) for a in sys.argv[1:]] or [HERE / n for n in DEFAULT]
    cache_path = HERE / "cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    w = load_weights()
    for path in paths:
        text = open(path, encoding="utf-8").read().splitlines()
        rows = rows_of(path, cache)
        prompts = [l.split("\t", 1)[1] for l in text]
        picks = {
            "tev1 choice": [a["tier"]["choice"] for _, a in rows],
            "router": [tier_for(score(features(p, a), w), w) for p, (_, a) in zip(prompts, rows)],
        }
        n = len(rows)
        for name, got in picks.items():
            want = [l for l, _ in rows]
            ok = sum(x == y for x, y in zip(want, got))
            miss = sum(x == "opus" and y != "opus" for x, y in zip(want, got))
            over = sum(x != "opus" and y == "opus" for x, y in zip(want, got))
            c = sum(cost(x, y) for x, y in zip(want, got))
            print(f"{path.name:<24}{name:<12} acc {ok}/{n} ({ok / n:.0%})  opus missed {miss}/"
                  f"{want.count('opus')}  sent to opus needlessly {over}/{n - want.count('opus')}  cost {c}")
    cache_path.write_text(json.dumps(cache))


if __name__ == "__main__":
    main()
