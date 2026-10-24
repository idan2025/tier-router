"""The keyword rules, with no tev1 and no network.

  python3 -I tests/test_signals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tier_router.classify import QUICK, SIGNALS

# (signal, text, should it fire?)
CASES = [
    ("hard", "the app deadlocks, find the root cause", True),
    ("hard", "תתכנן את הארכיטקטורה למעבר לריבוי אזורים", True),
    ("hard", "תעשה ביקורת אבטחה לכל בסיס הקוד", True),
    ("hard", "תוסיף עימוד ל-endpoint של /users", False),
    ("symptom", "why do some users get logged out randomly", True),
    ("symptom", "למה חלק מהמשתמשים מקבלים logout באופן אקראי", True),
    ("symptom", "ה-CPU קופץ ל-100% והשרת מפסיק להגיב, אין כלום בלוגים", True),
    ("symptom", "תוסיף מתג מצב כהה לעמוד ההגדרות", False),
    ("check", "is our token refresh flow safe against replay attacks", True),
    ("check", "האם תהליך רענון הטוקן שלנו בטוח?", True),
    ("check", "תכתוב בדיקות יחידה לפונקציה parse_duration", False),
    ("scope", "check everything before the release", True),
    ("scope", "תעבור על הכל לפני השחרור", True),
    ("scope", "תוסיף כפתור ייצוא", False),
    ("act", "fix it and then test it", True),
    ("act", "תתקן את הבאג ותריץ את הבדיקות", True),
    ("act", "תריץ את הבדיקות", False),
    ("act", "תתקן את הבאג ותבדוק שזה עובד", True),
]
QUICK_CASES = [
    ("push it", True),
    ("תעשה push", True),
    ("בסדר, תעשה merge", True),
    ("מה פורט ברירת המחדל של postgres?", True),
    ("תכתוב סקריפט שמגבה את התיקייה", False),
    ("design the sync protocol", False),
]


def main():
    bad = 0
    for name, text, want in CASES:
        got = bool(SIGNALS[name].search(text))
        if got != want:
            bad += 1
            print(f"BAD {name}: {text!r} fired={got}, wanted {want}")
    for text, want in QUICK_CASES:
        got = bool(QUICK.search(text))
        if got != want:
            bad += 1
            print(f"BAD quick: {text!r} matched={got}, wanted {want}")
    print("ALL OK" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
