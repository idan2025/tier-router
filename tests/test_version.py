"""/router shows the version in hooks/register.ts; it must be the one in plugin.json,
which is what Claude Code installs and compares when it updates the plugin.

  python3 -I tests/test_version.py
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())["version"]
shown = re.search(r"const VERSION = '([^']+)'", (ROOT / "hooks/register.ts").read_text()).group(1)
assert shown == manifest, f"register.ts says {shown}, plugin.json says {manifest}"
print(f"ok  /router shows {shown}, the installed version")
