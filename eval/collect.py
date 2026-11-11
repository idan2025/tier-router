"""Ask the decision model about every eval prompt once and cache its raw answers.

The cache is keyed by the exact state sent, so changing the questions or the
trimming re-asks only what changed. Each model has its own cache file: tev1's
is cache.json, laya's cache-laya.json (another tag, cache-<tag>.json), Jev's
cache-jev.json."""
import hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tier_router.classify import BACKEND, MODEL, QUESTIONS, ask_tev1, family, state_for

# laya is English-only: it is neither asked the Hebrew sets nor fitted on them.
ENGLISH_ONLY = BACKEND != "jev" and family(MODEL) == "laya"


def files(names):
    return [n for n in names if not (ENGLISH_ONLY and n.startswith("hebrew_"))]

HERE = Path(__file__).resolve().parent


def cache_path():
    if BACKEND == "jev":
        return HERE / "cache-jev.json"
    if family(MODEL) == "tev1":
        return HERE / "cache.json"
    tag = re.sub(r"[^A-Za-z0-9.-]+", "_", MODEL.removesuffix(":latest"))
    return HERE / f"cache-{tag}.json"


def load_cases():
    # real.jsonl holds prompts from the author's own sessions and is not
    # published; the evaluation runs on the rest without it.
    real = HERE / "real.jsonl"
    cases = [json.loads(l) for l in open(real)] if real.exists() else []
    for name in files(("synthetic.tsv", "hebrew_synthetic.tsv")):
        for line in open(HERE / name, encoding="utf-8"):
            label, prompt = line.rstrip("\n").split("\t", 1)
            cases.append({"prompt": prompt, "context": "", "label": label, "source": name.removesuffix(".tsv")})
    return cases


def load_blind():
    """Written after the first round of tuning and never fitted on."""
    cases = []
    for name in files(("blind.tsv", "hebrew_blind.tsv", "blind2.tsv", "hebrew_blind2.tsv", "blind3.tsv", "hebrew_blind3.tsv", "blind4.tsv", "hebrew_blind4.tsv")):
        for line in open(HERE / name, encoding="utf-8"):
            label, prompt = line.rstrip("\n").split("\t", 1)
            cases.append({"prompt": prompt, "context": "", "label": label, "source": name.removesuffix(".tsv")})
    return cases


def key(c):
    raw = json.dumps([QUESTIONS, state_for(c["prompt"], c.get("context", ""))], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def main():
    path = cache_path()
    cache = json.loads(path.read_text()) if path.exists() else {}
    cases = load_cases() + load_blind()
    asked = 0
    for c in cases:
        k = key(c)
        if k not in cache:
            cache[k] = ask_tev1(c["prompt"], c.get("context", ""), timeout=120)
            asked += 1
            if asked % 10 == 0:  # a stopped run keeps what it has asked so far
                path.write_text(json.dumps(cache))
    path.write_text(json.dumps(cache))
    print(f"{len(cases)} cases, {asked} asked, {len(cache)} cached")


if __name__ == "__main__":
    main()
