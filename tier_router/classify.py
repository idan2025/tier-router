"""Decide which Claude tier a request needs, with tev1 and a few rules.

tev1 is a decision model (Ollama's /v1/systemone), not a chat model: it gets
a short state and named questions, and returns a probability per option. Its
context is about 2,000 tokens, so the state is the end of the previous
assistant reply and the user's new message, both trimmed.

Four questions go in one call. Their answers, and a handful of keyword
signals, are folded into one difficulty score in [0, 1]; two thresholds turn
it into haiku, sonnet or opus. The weights and thresholds are fitted on
eval/ (see eval/tune.py) and live in weights.json.
"""

import json
import os
import re
import urllib.request
from pathlib import Path

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA
MODEL = os.environ.get("TEV1_MODEL", "tev1")
# The same question set can go to a cloud Jev instead (ROUTER_BACKEND=jev):
# TypeSafe's /v1/systemone takes the same body and answers in the same shape.
BACKEND = os.environ.get("ROUTER_BACKEND", "ollama")
JEV_URL = os.environ.get("TYPESAFE_URL", "https://api.typesafe.ai").rstrip("/")
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
TIERS = ("haiku", "sonnet", "opus")

QUESTIONS = {
    "tier": {
        "type": "choice",
        "instructions": "Which assistant should handle the user's new request? Pick the cheapest one that will do it well.",
        "criteria": {
            "haiku": "A quick answer from general knowledge, a status question, or one or two obvious commands or a tiny edit.",
            "sonnet": "A normal, well-defined coding task: build a feature, write a script or tests, fix a bug whose error is known.",
            "opus": "Hard work: find an unknown root cause, design or architect something, review a lot of code, or change a large system carefully.",
        },
    },
    "simple": {
        "type": "noul",
        "instructions": "Can the user's new request be done in a minute or two: answering a question, running one or two commands, or a tiny edit?",
    },
    "deep": {
        "type": "noul",
        "instructions": "Does the user's new request need investigating an unknown cause, designing something, reviewing a lot of code, or carefully changing many parts of a system?",
    },
    "effort": {
        "type": "score",
        "instructions": "How much work does the user's new request take?",
        "criteria": [
            "A quick answer or a single command.",
            "A small, clear change.",
            "A normal feature, script, or a bug with a clear error.",
            "A multi-step investigation or a large change.",
            "A hard open-ended problem: unknown root cause, architecture, security, or a big review.",
        ],
    },
}

# Keyword signals. Each only adds to the score with its fitted weight;
# none decides alone.
SIGNALS = {
    # Hard by name.
    "hard": re.compile(
        r"\b(design\w*|architect\w*|race|deadlock\w*|leak\w*|flak\w*|root cause|audit\w*|"
        r"air ?tight|production[- ]ready|investigat\w*|figure out|find out|track (it )?down|"
        r"reverse engineer\w*|prove|review (it|this|the|all|everything)|shard\w*|multi-region|encrypt\w*|zero downtime|"
        r"migrat\w* .{0,40}\b(across|whole|entire|monolith|database|every)\b)", re.I),
    # A failure described by its symptom, not its error: a hunt for a cause.
    "symptom": re.compile(
        r"\b(why|randomly|sometimes|intermittent\w*|stall\w*|hang\w*|desync\w*|spikes?|"
        r"(is|are|shows?|goes|went) down|broken|(does ?n[o']?t|never|not) "
        r"(work|go|reach|route|match|update|see|show|start)\w*|disagree|out of order|got worse|"
        r"can'?t reproduce|nothing in the logs|nothing obvious)\b", re.I),
    # Is it safe, secure, isolated, correct: an open-ended check.
    "check": re.compile(
        r"\b(safe|secure|security|isolat\w*|contained|vulnerab\w*|replay|check (if|whether|that|why|out)|"
        r"verify|is (it|this|our) (correct|right|ok))\b", re.I),
    # Everything at once.
    "scope": re.compile(
        r"\b(everything|whole|entire|all of (it|them)|fix all|before (the|we) (release|cut|ship)|"
        r"next cut|fully real|placeholder|anything (needs|else|that)|across \d+)\b", re.I),
    # More than one action in one instruction: work, not a one-liner.
    "act": re.compile(
        r"\b(and|then) (test|verify|release|deploy|implement|build|fix|run|check|update)\b", re.I),
}
# A quick one: a git step, a lookup, thanks. Only for a short message with
# no harder signal in it.
QUICK = re.compile(
    r"^\s*(yes,?\s*|ok,?\s*)?(please\s+)?(just\s+)?"
    r"(commit|push|pull|merge|tag|stage|open the pr|close the pr|run the tests|thanks|thank you|"
    r"rename|add the permission|install|show|list|print|what('?s| is| does| did)|how (do|many|big)|"
    r"is the|which|what branch)\b", re.I)

# The same rules in Hebrew: a straight translation of the word lists above.
# They are separate alternatives with no \b, because \b is ASCII-only in the
# TypeScript plugin and Hebrew words take prefixes (ה-, ב-, ל-, ו-) anyway.
# Spell ranges out ([א-ת]) rather than \w, so both languages match alike.
HEBREW = {
    "hard": (
        r"תכנון|תתכנן|ארכיטקטורה|דדלוק|דליפ|שורש הבעיה|ביקורת|אודיט|מוכן לפרודקשן|אטום|"
        r"תחקור|חקירה|תבין למה|תמצא למה|תמצא מאיפה|תמצא את (?:הסיבה|המקור)|תאתר|תגלה|"
        r"הנדסה לאחור|תוכיח|תעשה review|ריבוי אזורים|הצפנ|ללא downtime|בלי downtime|"
        r"(?:תעביר|מיגרציה|העברה).{0,40}(?:על פני|בכל|המונוליט|מסד הנתונים)"),
    "symptom": (
        r"למה|באקראי|אקראי|לפעמים|לסירוגין|נתקע|נתקעת|תקוע|קופא|קופץ|קפיצ|יוצא מסנכרון|"
        r"לא מסונכרן|נפל|נופל|לא זמין|שבור|נשבר|"
        r"לא (?:עובד|עובדת|עולה|מגיע|מגיעים|מנתב|תואמ|מתעדכן|רואה|רואים|מציג|מתחיל|מצליח)|אף פעם|"
        r"לא מסכימים|לא תואמ|לא לפי הסדר|שלא לפי הסדר|הורע|החמיר|לא מצליח[א-ת]* לשחזר|"
        r"אי אפשר לשחזר|אין כלום בלוגים|שום דבר בולט|שום דבר ברור"),
    "check": (
        r"בטוח|מאובטח|אבטחה|בידוד|מבודד|פגיעו|replay|תבדוק (?:אם|האם|ש|למה)|תוודא|לוודא|תאמת|"
        r"האם (?:זה|הוא|היא) (?:נכון|תקין)"),
    "scope": (
        r"הכל|כל ה(?:מערכת|קוד|בסיס|פרויקט)|כולו|כל מה ש|לפני השחרור|לפני שאנחנו משחררים|על פני \d+"),
    "act": r"(?:ו|ואז |אחר כך )(?:תבדוק|תריץ|תבנה|תתקן|תממש|תעדכן|תפרוס|תוודא)",
}
HEBREW_QUICK = (
    r"^\s*(?:(?:כן|בסדר|אוקיי)[,\s]*)?(?:בבקשה\s+)?(?:רק\s+)?"
    r"(?:תעשה (?:commit|push|pull|merge|stage|tag)|תמשוך|תריץ את הבדיקות|תודה|תשנה את השם|תתקין|"
    r"תציג|תראה|תדפיס|מה |כמה|איך |איזה|באיזה|על איזה|האם ה|תפתח את ה-PR|תסגור את ה-PR)")
SIGNALS = {n: re.compile(rx.pattern + "|" + HEBREW[n], re.I) for n, rx in SIGNALS.items()}
QUICK = re.compile(QUICK.pattern + "|" + HEBREW_QUICK, re.I)

WEIGHTS_FILE = Path(__file__).with_name("weights.json")
POSITIVE = ("tier", "deep", "effort", "hard", "symptom", "check", "scope", "act", "long")
NEGATIVE = ("simple", "quick")
DEFAULT_WEIGHTS = {
    "w_tier": 1.0, "w_deep": 0.5, "w_effort": 1.0, "w_hard": 0.5, "w_symptom": 0.5,
    "w_check": 0.5, "w_scope": 0.5, "w_act": 0.5, "w_long": 0.0,
    "w_simple": 0.5, "w_quick": 0.5,
    "t_sonnet": 0.35, "t_opus": 0.65,
}


def load_weights():
    try:
        return {**DEFAULT_WEIGHTS, **json.loads(WEIGHTS_FILE.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULT_WEIGHTS)


def _trim(text, limit):
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text or "", flags=re.S).strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + " … " + text[-(limit - head - 3):]


def state_for(prompt, context=""):
    parts = []
    if context:
        parts.append("End of the assistant's previous reply:\n" + _trim(context, 700))
    parts.append("The user's new request:\n" + _trim(prompt, 1800))
    return "\n\n".join(parts)


def ask_tev1(prompt, context="", timeout=30):
    """The decision model's raw answers for one request (local tev1 or cloud Jev)."""
    body = {"state": state_for(prompt, context), "questions": QUESTIONS}
    headers = {"Content-Type": "application/json"}
    if BACKEND == "jev":
        key = os.environ.get("TYPESAFE_API_KEY", "")
        if not key:
            raise RuntimeError("ROUTER_BACKEND=jev needs TYPESAFE_API_KEY")
        body["model"] = JEV_MODEL
        headers["Authorization"] = "Bearer " + key
        url = f"{JEV_URL}/v1/systemone"
    else:
        body["model"] = MODEL
        body["keep_alive"] = "30m"
        url = f"{OLLAMA}/v1/systemone"
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["answers"]


def features(prompt, answers):
    t = answers["tier"]["probabilities"]
    levels = len(QUESTIONS["effort"]["criteria"]) - 1
    text = _trim(prompt, 4000)
    f = {
        "tier": t.get("sonnet", 0) * 0.5 + t.get("opus", 0),
        "simple": answers["simple"]["noul"],
        "deep": answers["deep"]["noul"],
        "effort": answers["effort"].get("score", 0) / levels,
        "long": min(len(text) / 1500, 1.0),
    }
    for name, rx in SIGNALS.items():
        f[name] = min(len(rx.findall(text)), 2) / 2
    harder = any(f[n] for n in SIGNALS)
    f["quick"] = 1.0 if QUICK.search(text) and len(text) < 80 and not harder else 0.0
    return f


def score(f, w):
    """Difficulty in [0, 1]: 1 with every harder signal at full strength and
    no easier one, 0 the other way round."""
    pos = sum(w["w_" + n] * f[n] for n in POSITIVE)
    neg = sum(w["w_" + n] * f[n] for n in NEGATIVE)
    hi = sum(w["w_" + n] for n in POSITIVE)
    lo = -sum(w["w_" + n] for n in NEGATIVE)
    return (pos - neg - lo) / (hi - lo) if hi > lo else 0.5


def tier_for(s, w):
    if s >= w["t_opus"]:
        return "opus"
    if s >= w["t_sonnet"]:
        return "sonnet"
    return "haiku"


def classify(prompt, context="", weights=None):
    """(tier, score, features) for a request; raises if tev1 is unreachable."""
    w = weights or load_weights()
    f = features(prompt, ask_tev1(prompt, context))
    s = score(f, w)
    return tier_for(s, w), s, f
