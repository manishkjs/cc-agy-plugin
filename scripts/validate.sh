#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Missing required file: %s\n' "$1" >&2
    exit 1
  fi
}

for rel in \
  pyproject.toml \
  LICENSE \
  README.md \
  .mcp.json \
  .claude-plugin/plugin.json \
  .claude-plugin/marketplace.json \
  .agents/plugins/marketplace.json \
  plugins/cc-agy-plugin/.codex-plugin/plugin.json \
  plugins/cc-agy-plugin/LICENSE \
  plugins/cc-agy-plugin/skills/vertex/SKILL.md \
  skills/vertex/SKILL.md \
  scripts/install_agy_cli.sh \
  scripts/sync_docs.py \
  server/agy_cli.py \
  server/genai_client.py \
  server/verification_policy.py; do
  require_file "$root/$rel"
done

if ! cmp -s "$root/LICENSE" "$root/plugins/cc-agy-plugin/LICENSE"; then
  printf 'Codex plugin LICENSE does not match root LICENSE byte-for-byte.\n' >&2
  exit 1
fi

python3 - "$root" <<'PY'
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
m_ver = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
assert m_ver, "version missing in pyproject.toml"
ver = m_ver.group(1)

claude_plugin = json.loads((root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
claude_market = json.loads((root / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
codex_market = json.loads((root / ".agents" / "plugins" / "marketplace.json").read_text(encoding="utf-8"))
codex_plugin = json.loads((root / "plugins" / "cc-agy-plugin" / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
json.loads((root / ".mcp.json").read_text(encoding="utf-8"))

assert claude_plugin["version"] == ver, (claude_plugin["version"], ver)
assert claude_market["plugins"][0]["version"] == ver, (claude_market["plugins"][0]["version"], ver)
assert codex_plugin["version"] == ver, (codex_plugin["version"], ver)
assert codex_market["plugins"][0]["name"] == codex_plugin["name"] == "cc-agy-plugin"
assert codex_market["plugins"][0]["source"]["path"] == "./plugins/cc-agy-plugin"
for prompt in codex_plugin["interface"].get("defaultPrompt", []):
    assert len(prompt) <= 128, f"defaultPrompt exceeds 128 bytes: {prompt!r}"

readme = (root / "README.md").read_text(encoding="utf-8")
assert f"badge/version-{ver}-" in readme, f"README version badge out of sync with {ver}"
assert f"| **Version** | {ver}," in readme, f"README status version out of sync with {ver}"
assert not re.search(r"badge/tests-\d+%20passing", readme), "Avoid hardcoded test counts in README badge"
assert not re.search(r"\b\d+\s+automated(?:\s+unit,)?\s+tests\b", readme, re.I), (
    "Avoid hardcoded test counts in README prose"
)

sys.path.insert(0, str(root))
from server.harness_validator import validate_plugin_tree

tree_errors = validate_plugin_tree(root)
assert not tree_errors, "validate_plugin_tree failed:\n" + "\n".join(tree_errors)
PY

python3 "$root/scripts/sync_docs.py" --check --root "$root"

tmp_home="$(mktemp -d "${TMPDIR:-/tmp}/cc-agy-validate.XXXXXX")"
trap 'rm -rf "$tmp_home"' EXIT

install_json="$(
  cd "$root"
  CODEX_HOME="$tmp_home" GEMINI_API_KEY="test-key-not-written" \
    python3 -m server.install --host codex --profile balanced --base-url "https://gateway.example.internal" --max-daily-cents 500 --no-bootstrap --non-interactive --json || true
)"

python3 - "$install_json" <<'PY'
import json
import sys

report = json.loads(sys.argv[1])
assert report["preflight"]["auth_mode"] == "gateway", report["preflight"]
codex_host = next(h for h in report["hosts"] if h["host"] == "codex")
assert codex_host["status"] == "installed", codex_host
PY

test -f "$tmp_home/config.toml"
test -f "$tmp_home/skills/cc-agy-plugin/SKILL.md"
if grep -q "test-key-not-written" "$tmp_home/config.toml"; then
  printf 'API key leaked into config.toml!\n' >&2
  exit 1
fi

ent_home="$(mktemp -d "${TMPDIR:-/tmp}/cc-agy-validate-ent.XXXXXX")"
trap 'rm -rf "$tmp_home" "$ent_home"' EXIT

mkdir -p "$ent_home/bin" "$ent_home/agy-cfg"
cat > "$ent_home/bin/agy" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: agy [options]" >&2
  echo "  --output-format string" >&2
  exit 0
fi
exit 0
EOF
chmod +x "$ent_home/bin/agy"
printf 'oauth-token-placeholder-value\n' > "$ent_home/agy-cfg/antigravity-oauth-token"

ent_json="$(
  cd "$root"
  CODEX_HOME="$ent_home" \
  AGY_CONFIG_DIR="$ent_home/gemini-config" \
  CC_AGY_CLI_BIN="$ent_home/bin/agy" \
  CC_AGY_CLI_CONFIG_DIR="$ent_home/agy-cfg" \
  ANTIGRAVITY_OAUTH_TOKEN="tok-not-written" \
    python3 -m server.install --host codex --auth gemini-enterprise --profile balanced --no-bootstrap --non-interactive --json
)"

agy_host_json="$(
  cd "$root"
  AGY_CONFIG_DIR="$ent_home/gemini-config" \
  CC_AGY_CLI_BIN="$ent_home/bin/agy" \
  CC_AGY_CLI_CONFIG_DIR="$ent_home/agy-cfg" \
  ANTIGRAVITY_OAUTH_TOKEN="tok-not-written" \
    python3 -m server.install --host agy --auth gemini-enterprise --profile balanced --no-bootstrap --non-interactive --json
)"

python3 - "$ent_json" "$ent_home/config.toml" "$agy_host_json" "$ent_home/gemini-config/mcp_config.json" <<'PY'
from pathlib import Path
import json
import sys

report = json.loads(sys.argv[1])
assert report["preflight"]["auth_mode"] == "gemini_enterprise", report["preflight"]
assert report["preflight"]["ready"] is True, report["preflight"]
cfg_text = Path(sys.argv[2]).read_text(encoding="utf-8")
assert 'CC_AGY_AUTH_MODE = "gemini_enterprise"' in cfg_text, cfg_text
assert "tok-not-written" not in cfg_text, "Enterprise OAuth token leaked into config.toml!"

agy_report = json.loads(sys.argv[3])
agy_host = next(h for h in agy_report["hosts"] if h["host"] == "agy")
assert agy_host["status"] == "installed", agy_host
agy_mcp_text = Path(sys.argv[4]).read_text(encoding="utf-8")
agy_mcp = json.loads(agy_mcp_text)
assert agy_mcp["mcpServers"]["agy"]["env"]["CC_AGY_AUTH_MODE"] == "gemini_enterprise", agy_mcp
assert "tok-not-written" not in agy_mcp_text, "Enterprise OAuth token leaked into mcp_config.json!"
PY

printf 'Validation passed.\n'
