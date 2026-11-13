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
  package.json \
  LICENSE \
  README.md \
  .mcp.json \
  .claude-plugin/plugin.json \
  .claude-plugin/marketplace.json \
  .cursor-plugin/plugin.json \
  .cursor-plugin/mcp.json \
  .cursor-plugin/hooks.json \
  cursor/rules/cc-agy-coexistence.mdc \
  opencode-plugin/index.ts \
  .opencode/skills/cc-agy-plugin/SKILL.md \
  .opencode/agents/gemini-delegator.md \
  .opencode/commands/delegate.md \
  .opencode/commands/doctor.md \
  .opencode/commands/jobs.md \
  .opencode/commands/cost.md \
  .agents/plugins/marketplace.json \
  plugins/cc-agy-plugin/.codex-plugin/plugin.json \
  plugins/cc-agy-plugin/LICENSE \
  plugins/cc-agy-plugin/skills/vertex/SKILL.md \
  skills/vertex/SKILL.md \
  scripts/install_agy_cli.sh \
  scripts/quickstart.sh \
  scripts/lint_ast.py \
  scripts/run_tests_parallel.py \
  scripts/bootstrap-gateway.ps1 \
  scripts/sync_docs.py \
  server/acceptance_runner.py \
  server/agy_cli.py \
  server/agy_state.py \
  server/ast_regression_guards.py \
  server/bifrost_policy.py \
  server/env_flags.py \
  server/gateway.py \
  server/genai_client.py \
  server/guard_bash.py \
  server/guard_prompt_lint.py \
  server/guard_session_state.py \
  server/job_apply_mixin.py \
  server/job_git_utils.py \
  server/job_qa_mixin.py \
  server/job_salvage_mixin.py \
  server/rate_card.json \
  server/rate_card.py \
  server/settings_catalog.py \
  server/test_discovery.py \
  server/worker_prompt_builder.py \
  server/worker_sandbox.py \
  tests/fixtures/opencode_config_schema.json \
  server/telemetry.py \
  server/verification_policy.py; do
  require_file "$root/$rel"
done

python3 "$root/scripts/lint_ast.py"

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
cursor_plugin = json.loads((root / ".cursor-plugin" / "plugin.json").read_text(encoding="utf-8"))
cursor_mcp = json.loads((root / ".cursor-plugin" / "mcp.json").read_text(encoding="utf-8"))
cursor_hooks = json.loads((root / ".cursor-plugin" / "hooks.json").read_text(encoding="utf-8"))
codex_market = json.loads((root / ".agents" / "plugins" / "marketplace.json").read_text(encoding="utf-8"))
codex_plugin = json.loads((root / "plugins" / "cc-agy-plugin" / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8"))
package_json = json.loads((root / "package.json").read_text(encoding="utf-8"))
json.loads((root / ".mcp.json").read_text(encoding="utf-8"))

assert claude_plugin["version"] == ver, (claude_plugin["version"], ver)
assert claude_market["plugins"][0]["version"] == ver, (claude_market["plugins"][0]["version"], ver)
assert cursor_plugin["version"] == ver, (cursor_plugin["version"], ver)
assert package_json["version"] == ver, (package_json["version"], ver)
assert cursor_plugin["name"] == "cc-agy-plugin", cursor_plugin
assert "agy" in cursor_mcp.get("mcpServers", {}), cursor_mcp
assert isinstance(cursor_hooks.get("hooks"), dict), cursor_hooks
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

cursor_gw_json="$(
  cd "$root"
  CURSOR_CONFIG_DIR="$tmp_home/cursor" GEMINI_API_KEY="test-key-not-written" \
    python3 -m server.install --host cursor --profile balanced --base-url "https://gateway.example.internal" --max-daily-cents 500 --no-bootstrap --non-interactive --json || true
)"

python3 - "$cursor_gw_json" <<'PY'
import json
import sys

report = json.loads(sys.argv[1])
assert report["preflight"]["auth_mode"] == "gateway", report["preflight"]
cursor_host = next(h for h in report["hosts"] if h["host"] == "cursor")
assert cursor_host["status"] == "installed", cursor_host
PY

test -f "$tmp_home/cursor/mcp.json"
test -f "$tmp_home/cursor/skills/cc-agy-plugin/SKILL.md"
if grep -q "test-key-not-written" "$tmp_home/cursor/mcp.json"; then
  printf 'API key leaked into Cursor mcp.json!\n' >&2
  exit 1
fi

opencode_gw_json="$(
  cd "$root"
  OPENCODE_CONFIG_DIR="$tmp_home/opencode" GEMINI_API_KEY="test-key-not-written" \
    python3 -m server.install --host opencode --profile balanced --base-url "https://gateway.example.internal" --max-daily-cents 500 --no-bootstrap --non-interactive --json || true
)"

opencode_v2_json="$(
  cd "$root"
  OPENCODE_CONFIG_DIR="$tmp_home/opencode_v2" GEMINI_API_KEY="test-key-not-written" \
    python3 -m server.install --host opencode --opencode-v2 --opencode-codemode --opencode-protocol auto --profile balanced --base-url "https://gateway.example.internal" --max-daily-cents 500 --no-bootstrap --non-interactive --json || true
)"

python3 - "$opencode_gw_json" "$tmp_home/opencode/opencode.json" "$opencode_v2_json" "$tmp_home/opencode_v2/opencode.json" "$root/tests/fixtures/opencode_config_schema.json" <<'PY'
from pathlib import Path
import json
import os
import sys

report = json.loads(sys.argv[1])
assert report["preflight"]["auth_mode"] == "gateway", report["preflight"]
opencode_host = next(h for h in report["hosts"] if h["host"] == "opencode")
assert opencode_host["status"] == "installed", opencode_host
oc_cfg = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
assert oc_cfg["$schema"] == "https://opencode.ai/config.json", oc_cfg
assert oc_cfg["mcp"]["agy"]["type"] == "local", oc_cfg
assert oc_cfg["mcp"]["agy"]["enabled"] is True, oc_cfg
assert oc_cfg["mcp"]["agy"]["timeout"] == 1800000, oc_cfg

report_v2 = json.loads(sys.argv[3])
opencode_v2_host = next(h for h in report_v2["hosts"] if h["host"] == "opencode")
assert opencode_v2_host["status"] == "installed", opencode_v2_host
oc_v2_cfg = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
assert oc_v2_cfg["$schema"] == "https://opencode.ai/config.json", oc_v2_cfg
assert oc_v2_cfg["mcp"]["servers"]["agy"]["type"] == "local", oc_v2_cfg
assert oc_v2_cfg["mcp"]["servers"]["agy"]["disabled"] is False, oc_v2_cfg
assert oc_v2_cfg["mcp"]["servers"]["agy"]["pool"] is False, oc_v2_cfg
assert oc_v2_cfg["mcp"]["servers"]["agy"]["codemode"] is True, oc_v2_cfg
assert oc_v2_cfg["mcp"]["servers"]["agy"]["protocol"] == "auto", oc_v2_cfg
assert oc_v2_cfg["mcp"]["servers"]["agy"]["timeout"] == {"startup": 30000, "execution": 1800000}, oc_v2_cfg

try:
    import jsonschema
except ImportError:
    jsonschema = None
if jsonschema is None:
    if os.environ.get("CI", "").strip().lower() in ("1", "true", "yes"):
        raise SystemExit("jsonschema must be installed in CI for OpenCode JSON Schema validation")
    print("jsonschema not installed; skipping OpenCode JSON Schema validation", file=sys.stderr)
else:
    schema = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
    validator = jsonschema.validators.validator_for(schema)(schema)
    for label, candidate in (("v1", oc_cfg), ("v2", oc_v2_cfg)):
        errs = [e.message for e in validator.iter_errors(candidate)]
        assert not errs, f"OpenCode {label} config failed schema validation: {errs}"
PY

bad_proto_rc=0
bad_proto_out="$(
  cd "$root" && OPENCODE_CONFIG_DIR="$tmp_home/opencode_bad_proto" \
    python3 -m server.install --host opencode --opencode-v2 --opencode-protocol 2026-07-28 --no-bootstrap --non-interactive 2>&1
)" || bad_proto_rc=$?
if [[ "$bad_proto_rc" -ne 2 || "$bad_proto_out" != *"invalid choice"* ]]; then
  printf 'Expected exit 2 + invalid choice for --opencode-protocol 2026-07-28, got rc=%d out=%s\n' "$bad_proto_rc" "$bad_proto_out" >&2
  exit 1
fi

test -f "$tmp_home/opencode/opencode.json"
test -f "$tmp_home/opencode/skills/cc-agy-plugin/SKILL.md"
test -f "$tmp_home/opencode/commands/doctor.md"
test -f "$tmp_home/opencode/commands/delegate.md"
test -f "$tmp_home/opencode/commands/jobs.md"
test -f "$tmp_home/opencode/commands/cost.md"
test -f "$tmp_home/opencode/agents/gemini-delegator.md"
if grep -q "test-key-not-written" "$tmp_home/opencode/opencode.json" "$tmp_home/opencode_v2/opencode.json"; then
  printf 'API key leaked into OpenCode opencode.json!\n' >&2
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

cursor_ent_json="$(
  cd "$root"
  CURSOR_CONFIG_DIR="$ent_home/cursor" \
  CC_AGY_CLI_BIN="$ent_home/bin/agy" \
  CC_AGY_CLI_CONFIG_DIR="$ent_home/agy-cfg" \
  ANTIGRAVITY_OAUTH_TOKEN="tok-not-written" \
    python3 -m server.install --host cursor --auth gemini-enterprise --profile balanced --no-bootstrap --non-interactive --json
)"

opencode_ent_json="$(
  cd "$root"
  OPENCODE_CONFIG_DIR="$ent_home/opencode" \
  CC_AGY_CLI_BIN="$ent_home/bin/agy" \
  CC_AGY_CLI_CONFIG_DIR="$ent_home/agy-cfg" \
  ANTIGRAVITY_OAUTH_TOKEN="tok-not-written" \
    python3 -m server.install --host opencode --auth gemini-enterprise --profile balanced --no-bootstrap --non-interactive --json
)"

python3 - "$ent_json" "$ent_home/config.toml" "$agy_host_json" "$ent_home/gemini-config/mcp_config.json" "$cursor_ent_json" "$ent_home/cursor/mcp.json" "$opencode_ent_json" "$ent_home/opencode/opencode.json" <<'PY'
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

cursor_report = json.loads(sys.argv[5])
cursor_host = next(h for h in cursor_report["hosts"] if h["host"] == "cursor")
assert cursor_host["status"] == "installed", cursor_host
cursor_mcp_text = Path(sys.argv[6]).read_text(encoding="utf-8")
cursor_mcp = json.loads(cursor_mcp_text)
assert cursor_mcp["mcpServers"]["agy"]["env"]["CC_AGY_AUTH_MODE"] == "gemini_enterprise", cursor_mcp
assert "tok-not-written" not in cursor_mcp_text, "Enterprise OAuth token leaked into Cursor mcp.json!"

opencode_report = json.loads(sys.argv[7])
opencode_host = next(h for h in opencode_report["hosts"] if h["host"] == "opencode")
assert opencode_host["status"] == "installed", opencode_host
opencode_mcp_text = Path(sys.argv[8]).read_text(encoding="utf-8")
opencode_mcp = json.loads(opencode_mcp_text)
assert opencode_mcp["mcp"]["agy"]["environment"]["CC_AGY_AUTH_MODE"] == "gemini_enterprise", opencode_mcp
assert "tok-not-written" not in opencode_mcp_text, "Enterprise OAuth token leaked into OpenCode opencode.json!"
PY

printf 'Validation passed.\n'
