"""Repository-agnostic unit tests for Iteration 6 plugin fixes (Items A, B, C, and D)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.post_tool_ledger import evaluate_post_tool_use
from scripts.pre_tool_guard import _lead_review_read_bonus
from scripts.session_start import (
    REVIEW_AFTER_APPLY_GUIDANCE,
    REVIEW_BEFORE_APPLY_GUIDANCE,
)
from server.agy_runner import (
    WORKER_SYSTEM_PROMPT,
    build_worker_system_prompt,
    resolve_thinking_level,
)
from server.jobs import (
    JobManager,
    detect_class_attribute_shadowing,
)
from server.receipts import ReceiptStore
from server.regions import estimate_tokens


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True)


# ---------------------------------------------------------------------------
# Item A: Pre-Flight Clean-HEAD Baseline Acceptance Probe & Failure Subtraction
# ---------------------------------------------------------------------------


def test_item_a_baseline_cwd_probe_hygiene_and_post_worker_subtraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CC_AGY_STRICT_MODE", "true")
    repo = tmp_path / "repo_cwd"
    _init_repo(repo)

    (repo / ".gitignore").write_text("ignored_probe_cache/\n", encoding="utf-8")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "pipeline.py").write_text(
        "from pathlib import Path\n\n"
        "class Pipeline:\n"
        "    artifact_dir = Path('default_artifacts')\n\n"
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n",
        encoding="utf-8",
    )
    (repo / "tracked_seed.txt").write_text("clean-seed\n", encoding="utf-8")

    # Legacy test in a subdirectory that assumes cwd == tests/test_legacy and pollutes disk when run from root
    legacy_dir = repo / "tests" / "test_legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "local_fixture.txt").write_text("ok-fixture\n", encoding="utf-8")
    (legacy_dir / "test_legacy_cwd.py").write_text(
        "from pathlib import Path\n\n"
        "def test_reads_relative_fixture():\n"
        "    # Side effects if run from repo root: untracked dir, gitignored dir, and tracked file mutation\n"
        "    if Path('tracked_seed.txt').exists():\n"
        "        Path('tracked_seed.txt').write_text('mutated-by-probe\\n', encoding='utf-8')\n"
        "        Path('probe_leftover_dir').mkdir(exist_ok=True)\n"
        "        Path('ignored_probe_cache').mkdir(exist_ok=True)\n"
        "        (Path('ignored_probe_cache') / 'junk.bin').write_text('x', encoding='utf-8')\n"
        "    data = Path('local_fixture.txt').read_text(encoding='utf-8').strip()\n"
        "    assert data == 'ok-fixture'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "initial brownfield repo"], cwd=str(repo), check=True)

    captured_prompts: list[str] = []
    worktree_state_at_worker_start: dict[str, Any] = {}

    def clean_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        captured_prompts.extend(req["prompts"])
        status_out = subprocess.run(
            ["git", "status", "--porcelain", "--ignored"],
            cwd=str(wt),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        worktree_state_at_worker_start["git_status_ignored"] = status_out
        worktree_state_at_worker_start["seed_content"] = (wt / "tracked_seed.txt").read_text(encoding="utf-8")
        worktree_state_at_worker_start["leftover_dir_exists"] = (wt / "probe_leftover_dir").exists()
        worktree_state_at_worker_start["ignored_dir_exists"] = (wt / "ignored_probe_cache").exists()

        # Implement requested feature cleanly WITHOUT shadowing Pipeline.artifact_dir in __init__
        (wt / "pkg" / "pipeline.py").write_text(
            "import json\n"
            "from pathlib import Path\n\n"
            "class Pipeline:\n"
            "    artifact_dir = Path('default_artifacts')\n\n"
            "    def __init__(self, name: str) -> None:\n"
            "        self.name = name\n\n"
            "    def save_manifest(self, columns: list[str]) -> Path:\n"
            "        out_dir = Path(self.artifact_dir)\n"
            "        out_dir.mkdir(parents=True, exist_ok=True)\n"
            "        dest = out_dir / 'manifest.json'\n"
            "        dest.write_text(json.dumps({'name': self.name, 'columns': columns}), encoding='utf-8')\n"
            "        return dest\n",
            encoding="utf-8",
        )
        (wt / "tests" / "test_manifest.py").write_text(
            "import json\n"
            "from pkg.pipeline import Pipeline\n\n"
            "def test_save_manifest_respects_class_monkeypatch(tmp_path, monkeypatch):\n"
            "    monkeypatch.setattr(Pipeline, 'artifact_dir', tmp_path)\n"
            "    p = Pipeline('demo')\n"
            "    dest = p.save_manifest(['a', 'b'])\n"
            "    assert dest == tmp_path / 'manifest.json'\n"
            "    assert json.loads(dest.read_text(encoding='utf-8')) == {'name': 'demo', 'columns': ['a', 'b']}\n",
            encoding="utf-8",
        )
        return {
            "text": "Implemented save_manifest and unit tests.",
            "usage": {"prompt_token_count": 1000, "candidates_token_count": 200, "total_token_count": 1200},
        }

    store = ReceiptStore(tmp_path / "data_cwd", workspace_root=repo)
    jm = JobManager(workspace_root=repo, store=store, worker_backend=clean_worker)
    try:
        digest = jm.delegate_start(
            "Add save_manifest(columns) to Pipeline in pkg/pipeline.py.",
            project="p",
            acceptance=["pytest -x -q tests"],
            wait=True,
        )
        # 1. Pre-flight probe left 100% clean worktree for the worker
        assert worktree_state_at_worker_start["seed_content"] == "clean-seed\n"
        assert worktree_state_at_worker_start["leftover_dir_exists"] is False
        assert worktree_state_at_worker_start["ignored_dir_exists"] is False
        assert "probe_leftover_dir" not in worktree_state_at_worker_start["git_status_ignored"]
        assert "ignored_probe_cache" not in worktree_state_at_worker_start["git_status_ignored"]

        # 2. Worker prompt injected [Baseline Acceptance Note] and softened Acceptance wording
        assert len(captured_prompts) == 1
        wp = captured_prompts[0]
        assert "[Baseline Acceptance Note]" in wp
        assert "tests/test_legacy" in wp
        assert "NEVER modify or shadow existing production class/module attributes" in wp
        assert "you may inspect how an existing test file sets up fixtures or `cwd` when needed" in wp

        # 3. Post-worker acceptance passed via subdir + remaining-suite verification and auto-applied
        assert digest["status"] == "ok"
        assert digest["acceptance_passed"] is True
        assert digest["applied"] is True
        assert digest.get("baseline_acceptance_warnings")

        rec = store.read_receipt(digest["job_id"])
        assert rec.get("baseline_acceptance_probe")
        probe0 = rec["baseline_acceptance_probe"][0]
        assert probe0["subdir_cwd_passed"] is True
        assert probe0["passing_cwd"] == "tests/test_legacy"
        acc0 = rec["acceptance_results"][0]
        assert acc0["passed"] is True
        assert acc0["preexisting_baseline_failure_ignored"] is True
        assert acc0["counts"]["passed"] == 2
    finally:
        jm.shutdown()


def test_item_a_does_not_mask_failing_new_tests_or_unfixed_tdd_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CC_AGY_STRICT_MODE", "true")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "")

    # 1. Subdir cwd quirk exists, PLUS worker's newly added test genuinely fails -> MUST FAIL acceptance!
    repo1 = tmp_path / "repo_new_fail"
    _init_repo(repo1)
    (repo1 / "pkg").mkdir()
    (repo1 / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo1 / "pkg" / "service.py").write_text("def compute(x: int) -> int:\n    return x\n", encoding="utf-8")
    sub = repo1 / "tests" / "test_sub"
    sub.mkdir(parents=True)
    (sub / "local.txt").write_text("hello\n", encoding="utf-8")
    (sub / "test_sub.py").write_text(
        "from pathlib import Path\n"
        "def test_local():\n"
        "    assert Path('local.txt').read_text(encoding='utf-8').strip() == 'hello'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo1), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo1), check=True)

    def broken_feature_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "pkg" / "service.py").write_text("def compute(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
        (wt / "tests" / "test_new_service.py").write_text(
            "from pkg.service import compute\n"
            "def test_compute_double():\n"
            "    assert compute(5) == 10\n",
            encoding="utf-8",
        )
        return {"text": "Attempted feature.", "usage": {"prompt_token_count": 500, "total_token_count": 600}}

    store1 = ReceiptStore(tmp_path / "data_new_fail", workspace_root=repo1)
    jm1 = JobManager(workspace_root=repo1, store=store1, worker_backend=broken_feature_worker)
    try:
        dig1 = jm1.delegate_start(
            "Update compute in pkg/service.py",
            project="p",
            acceptance=["pytest -x -q tests"],
            auto_apply=False,
            wait=True,
        )
        assert dig1["acceptance_passed"] is False
        rec1 = store1.read_receipt(dig1["job_id"])
        assert not rec1["acceptance_results"][0].get("preexisting_baseline_failure_ignored")
    finally:
        jm1.shutdown()

    # 2. Classic TDD acceptance test (fails on HEAD in every directory) left broken by worker -> MUST FAIL!
    repo2 = tmp_path / "repo_tdd"
    _init_repo(repo2)
    (repo2 / "calc.py").write_text("def add(a: int, b: int) -> int:\n    return 0\n", encoding="utf-8")
    (repo2 / "test_calc.py").write_text(
        "from calc import add\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo2), check=True)
    subprocess.run(["git", "commit", "-qm", "init tdd"], cwd=str(repo2), check=True)

    def wrong_tdd_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "calc.py").write_text("def add(a: int, b: int) -> int:\n    return 4\n", encoding="utf-8")
        return {"text": "Edited calc.", "usage": {"prompt_token_count": 400, "total_token_count": 450}}

    store2 = ReceiptStore(tmp_path / "data_tdd", workspace_root=repo2)
    jm2 = JobManager(workspace_root=repo2, store=store2, worker_backend=wrong_tdd_worker)
    try:
        dig2 = jm2.delegate_start(
            "Fix add(a, b) in calc.py",
            project="p",
            acceptance=["pytest -q test_calc.py"],
            auto_apply=False,
            wait=True,
        )
        assert dig2["acceptance_passed"] is False
    finally:
        jm2.shutdown()


# ---------------------------------------------------------------------------
# Item B: Brownfield Class/Module Attribute Contract Invariant (<350 tokens)
# ---------------------------------------------------------------------------


def test_item_b_worker_system_prompt_contract_invariant_and_token_budget() -> None:
    for profile in ("full", "command-output", "read-only"):
        prompt = build_worker_system_prompt("/tmp/ws", profile=profile, has_report_to_lead=True)
        assert estimate_tokens(prompt) < 350, f"profile={profile} exceeded 350 tokens: {estimate_tokens(prompt)}"
        assert "monkeypatch.setattr" in prompt
        assert "tmp_path" in prompt
        assert "__init__" in prompt
    assert estimate_tokens(WORKER_SYSTEM_PROMPT) < 350
    # Default Flash coding thinking_level remains LOW (Suggestion F ignored)
    assert resolve_thinking_level(None, model="gemini-3.8-flash", read_only=False) == "LOW"


# ---------------------------------------------------------------------------
# Item C: Preserve User Spec, Checklist, Plan & Spec/Context Files on Escalation
# ---------------------------------------------------------------------------


def test_item_c_plugin_escalation_preserves_full_task_specification_and_checklist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CC_AGY_STRICT_MODE", "true")
    monkeypatch.setenv("CC_AGY_AUTO_CONTINUE_DIRTY_WORKTREE", "1")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "gemini-4-argon")

    repo = tmp_path / "repo_esc_ctx"
    _init_repo(repo)
    (repo / "SPEC.md").write_text("# Detailed Spec\nMust emit `schema_version = 2` in JSON output.\n", encoding="utf-8")
    (repo / "exporter.py").write_text(
        "class Exporter:\n"
        "    def dump(self) -> dict:\n"
        "        return {}\n",
        encoding="utf-8",
    )
    (repo / "test_exporter.py").write_text(
        "from exporter import Exporter\n"
        "def test_dump():\n"
        "    out = Exporter().dump()\n"
        "    assert out == {'schema_version': 2, 'ok': True}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    store = ReceiptStore(tmp_path / "data_esc_ctx", workspace_root=repo)
    now_ts = time.time()
    (store.data_dir / ".lead_session_started").write_text(
        json.dumps({"session_id": "sess-iter6", "ts": now_ts - 1.0}),
        encoding="utf-8",
    )
    full_user_spec = (
        "Update Exporter.dump() in exporter.py so that it returns {'schema_version': 2, 'ok': True}, "
        "preserving backward compatibility with existing callers and verifying all edge cases in test_exporter.py. "
        "Ensure deterministic keys and zero side effects on import."
    )
    (store.data_dir / ".last_user_prompt.json").write_text(
        json.dumps({"session_id": "sess-iter6", "ts": now_ts, "prompt": full_user_spec}),
        encoding="utf-8",
    )

    # First test auto-attached raw_user_prompt + checklist + plan + context_files (no spec_files)
    seen_calls: list[tuple[str, str]] = []

    def two_phase_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        model_name = req["model"]
        prompt_text = req["prompts"][0]
        seen_calls.append((model_name, prompt_text))
        if model_name == "gemini-3.8-flash":
            # Partial edit that hits checkpoint before completing implementation
            (wt / "exporter.py").write_text(
                "class Exporter:\n"
                "    def dump(self) -> dict:\n"
                "        return {'schema_version': 2}\n",
                encoding="utf-8",
            )
            return {
                "text": "Partial edit before tool cap.",
                "stop_reason": "MAX_TOOL_CALLS_EXCEEDED",
                "usage": {"prompt_token_count": 2000, "candidates_token_count": 300, "total_token_count": 2300},
            }
        # Escalated Argon turn fixes exporter.py to pass acceptance
        (wt / "exporter.py").write_text(
            "class Exporter:\n"
            "    def dump(self) -> dict:\n"
            "        return {'schema_version': 2, 'ok': True}\n",
            encoding="utf-8",
        )
        return {
            "text": "Completed Exporter.dump implementation.",
            "usage": {"prompt_token_count": 2500, "candidates_token_count": 250, "total_token_count": 2750},
        }

    jm = JobManager(workspace_root=repo, store=store, worker_backend=two_phase_worker)
    try:
        digest = jm.delegate_start(
            "Implement attached Exporter spec.",
            model="gemini-3.8-flash",
            project="p",
            checklist=[
                "Return schema_version=2 and ok=True from Exporter.dump()",
                "Ensure test_exporter.py passes",
            ],
            plan=["Update Exporter.dump in exporter.py", "Run pytest -q test_exporter.py"],
            context_files=["exporter.py"],
            acceptance=["pytest -q test_exporter.py"],
            wait=True,
        )
        assert [m for m, _ in seen_calls] == ["gemini-3.8-flash", "gemini-4-argon"]
        argon_prompt = seen_calls[1][1]
        assert "## Original User Request" in argon_prompt
        assert "returns {'schema_version': 2, 'ok': True}" in argon_prompt
        assert "## Behavioral Checklist" in argon_prompt
        assert "Return schema_version=2 and ok=True from Exporter.dump()" in argon_prompt
        assert "## Plan from the lead" in argon_prompt
        assert "## Pre-Read Context Files" in argon_prompt
        # Context file in Argon prompt reflects Flash's partial worktree state rather than stale pre-Flash file!
        assert "return {'schema_version': 2}" in argon_prompt
        assert digest["status"] == "ok"
        assert digest["acceptance_passed"] is True
        assert digest["applied"] is True
        assert digest["escalated_from_model"] == "gemini-3.8-flash"
    finally:
        jm.shutdown()


# ---------------------------------------------------------------------------
# Item D: Zero-Token AST Class-Attribute Shadowing Detector + Conditional Audit
# ---------------------------------------------------------------------------


def test_item_d_ast_detector_catches_init_and_helper_shadowing_without_false_positives(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo_ast"
    _init_repo(repo)
    (repo / "pkg").mkdir()
    (repo / "pkg" / "runner.py").write_text(
        "from pathlib import Path\n\n"
        "class Runner:\n"
        "    output_root = Path('outputs')\n"
        "    timeout_sec: int = 30\n"
        "    already_instance = 'default'\n\n"
        "    @property\n"
        "    def computed(self) -> str:\n"
        "        return 'x'\n\n"
        "    def __init__(self, label: str) -> None:\n"
        "        self.label = label\n"
        "        self.already_instance = 'instance_override'\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_runner.py").write_text(
        "class TestHelper:\n    flag = True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=str(repo), check=True)

    # 1. Shadowing output_root directly in __init__ AND timeout_sec via helper called from __init__
    (repo / "pkg" / "runner.py").write_text(
        "from pathlib import Path\n\n"
        "class Runner:\n"
        "    output_root = Path('outputs')\n"
        "    timeout_sec: int = 30\n"
        "    already_instance = 'default'\n\n"
        "    @property\n"
        "    def computed(self) -> str:\n"
        "        return 'x'\n\n"
        "    def __init__(self, label: str) -> None:\n"
        "        self.label = label\n"
        "        self.already_instance = 'updated_instance'\n"
        "        self.output_root = Path.cwd() / 'outputs'\n"
        "        self._init_runtime()\n\n"
        "    def _init_runtime(self) -> None:\n"
        "        self.timeout_sec = 60\n",
        encoding="utf-8",
    )
    # Test files must never be flagged
    (repo / "tests" / "test_runner.py").write_text(
        "class TestHelper:\n"
        "    flag = True\n"
        "    def __init__(self):\n"
        "        self.flag = False\n",
        encoding="utf-8",
    )

    warnings = detect_class_attribute_shadowing(
        repo,
        ["pkg/runner.py", "tests/test_runner.py"],
        baseline_ref="HEAD",
    )
    assert len(warnings) == 2
    assert any("Runner.__init__ newly assigns self.output_root" in w for w in warnings)
    assert any("Runner._init_runtime (called from __init__) newly assigns self.timeout_sec" in w for w in warnings)
    assert not any("already_instance" in w for w in warnings)
    assert not any("TestHelper" in w for w in warnings)

    # 2. Explicit optional caller guard (`if output_root is not None: self.output_root = output_root`) is NOT flagged
    (repo / "pkg" / "runner.py").write_text(
        "from pathlib import Path\n\n"
        "class Runner:\n"
        "    output_root = Path('outputs')\n"
        "    timeout_sec: int = 30\n"
        "    already_instance = 'default'\n\n"
        "    def __init__(self, label: str, output_root: Path | None = None) -> None:\n"
        "        self.label = label\n"
        "        self.already_instance = 'instance_override'\n"
        "        if output_root is not None:\n"
        "            self.output_root = output_root\n",
        encoding="utf-8",
    )
    assert detect_class_attribute_shadowing(repo, ["pkg/runner.py"], baseline_ref="HEAD") == []


def test_item_d_digest_hooks_and_lead_guidance_gate_on_contract_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CC_AGY_STRICT_MODE", "true")
    repo = tmp_path / "repo_gate"
    _init_repo(repo)
    (repo / "service.py").write_text(
        "from pathlib import Path\n\n"
        "class Client:\n"
        "    cache_dir = Path('cache')\n\n"
        "    def __init__(self, endpoint: str) -> None:\n"
        "        self.endpoint = endpoint\n",
        encoding="utf-8",
    )
    (repo / "test_service.py").write_text(
        "from service import Client\n"
        "def test_client():\n"
        "    c = Client('https://example.test')\n"
        "    assert c.endpoint == 'https://example.test'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)

    # Guidance text check
    assert "contract_warnings is empty" in REVIEW_AFTER_APPLY_GUIDANCE
    assert "get_diff(focus_symbols=[...])" in REVIEW_AFTER_APPLY_GUIDANCE
    assert "get_diff(focus_symbols=[...])" in REVIEW_BEFORE_APPLY_GUIDANCE

    def shadowing_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "service.py").write_text(
            "from pathlib import Path\n\n"
            "class Client:\n"
            "    cache_dir = Path('cache')\n\n"
            "    def __init__(self, endpoint: str) -> None:\n"
            "        self.endpoint = endpoint\n"
            "        self.cache_dir = Path('/tmp/override')\n",
            encoding="utf-8",
        )
        return {"text": "Updated Client.", "usage": {"prompt_token_count": 800, "total_token_count": 900}}

    store = ReceiptStore(tmp_path / "data_gate", workspace_root=repo)
    (store.data_dir / ".lead_session_started").write_text(
        json.dumps({"session_id": "s-gate", "ts": time.time() - 1.0}),
        encoding="utf-8",
    )
    jm = JobManager(workspace_root=repo, store=store, worker_backend=shadowing_worker)
    try:
        digest = jm.delegate_start(
            "Update Client in service.py",
            project="p",
            acceptance=["pytest -q test_service.py"],
            wait=True,
        )
        assert digest["applied"] is True
        assert digest["acceptance_passed"] is True
        assert digest.get("contract_warnings")
        assert "Client.__init__ newly assigns self.cache_dir" in digest["contract_warnings"][0]
        assert "get_diff" in digest["next_action"]
        assert "focus_symbols" in digest["next_action"]

        # PostToolUse ledger must NOT force immediate zero-tool-call stop when contract_warnings is non-empty
        post_dec = evaluate_post_tool_use(
            {
                "cwd": str(repo),
                "tool_name": "mcp__plugin_cc-agy_cc-agy__delegate_start",
                "tool_response": digest,
            },
            env={"CC_AGY_STRICT_MODE": "true"},
            store=store,
        )
        assert post_dec == {}
        assert _lead_review_read_bonus(store.data_dir, {"CC_AGY_STRICT_MODE": "true"}) == 6
    finally:
        jm.shutdown()

    # Clean worker run (zero contract_warnings) keeps 0-cent immediate exit behavior
    store_clean = ReceiptStore(tmp_path / "data_clean", workspace_root=repo)
    (store_clean.data_dir / ".lead_session_started").write_text(
        json.dumps({"session_id": "s-clean", "ts": time.time() - 1.0}),
        encoding="utf-8",
    )
    subprocess.run(["git", "checkout", "--", "."], cwd=str(repo), check=True)

    def clean_client_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "service.py").write_text(
            "from pathlib import Path\n\n"
            "class Client:\n"
            "    cache_dir = Path('cache')\n\n"
            "    def __init__(self, endpoint: str) -> None:\n"
            "        self.endpoint = endpoint\n\n"
            "    def ping(self) -> bool:\n"
            "        return True\n",
            encoding="utf-8",
        )
        return {"text": "Added ping().", "usage": {"prompt_token_count": 700, "total_token_count": 800}}

    jm_clean = JobManager(workspace_root=repo, store=store_clean, worker_backend=clean_client_worker)
    try:
        dig_clean = jm_clean.delegate_start(
            "Add ping() to Client in service.py",
            project="p",
            acceptance=["pytest -q test_service.py"],
            wait=True,
        )
        assert dig_clean["applied"] is True
        assert "contract_warnings" not in dig_clean
        assert "Finish now in one concise sentence" in dig_clean["next_action"]
        post_clean = evaluate_post_tool_use(
            {
                "cwd": str(repo),
                "tool_name": "mcp__plugin_cc-agy_cc-agy__delegate_start",
                "tool_response": dig_clean,
            },
            env={"CC_AGY_STRICT_MODE": "true"},
            store=store_clean,
        )
        assert "finish immediately in one concise sentence" in post_clean["hookSpecificOutput"]["additionalContext"]
        assert _lead_review_read_bonus(store_clean.data_dir, {"CC_AGY_STRICT_MODE": "true"}) == 0
    finally:
        jm_clean.shutdown()


def test_opus_round1_regressions_tdd_false_pass_scoped_prompt_and_ast_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover all 4 adversarial edge cases found by Claude Opus 5.5 (High) in Round 1."""
    monkeypatch.setenv("CC_AGY_STRICT_MODE", "true")
    monkeypatch.setenv("CC_AGY_ESCALATE_MODEL_ON_FAILURE", "")
    monkeypatch.setenv("CC_AGY_AUTO_APPLY_ON_PASS", "1")

    # 1. TDD test failing at HEAD MUST NOT auto-apply if worker leaves it broken and adds an unrelated passing test
    repo_tdd = tmp_path / "repo_tdd_extra"
    _init_repo(repo_tdd)
    (repo_tdd / "calc.py").write_text("def add(a: int, b: int) -> int:\n    return 0\n", encoding="utf-8")
    (repo_tdd / "tests").mkdir()
    (repo_tdd / "tests" / "conftest.py").write_text(
        "import pathlib, sys\nsys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n",
        encoding="utf-8",
    )
    (repo_tdd / "tests" / "test_calc.py").write_text(
        "from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_tdd), check=True)
    subprocess.run(["git", "commit", "-qm", "init tdd"], cwd=str(repo_tdd), check=True)

    def lazy_tdd_worker(req: dict[str, Any]) -> dict[str, Any]:
        wt = Path(req["workspace_dir"])
        (wt / "calc.py").write_text("def add(a: int, b: int) -> int:\n    return 4\n", encoding="utf-8")
        (wt / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
        return {"text": "done", "usage": {"prompt_token_count": 100, "total_token_count": 120}}

    st_tdd = ReceiptStore(tmp_path / "data_tdd_extra", workspace_root=repo_tdd)
    jm_tdd = JobManager(workspace_root=repo_tdd, store=st_tdd, worker_backend=lazy_tdd_worker)
    try:
        dig_tdd = jm_tdd.delegate_start(
            "Fix add(a, b) in calc.py so tests/test_calc.py passes",
            project="p",
            acceptance=["python3 -m pytest -q tests"],
            wait=True,
        )
        assert dig_tdd["acceptance_passed"] is False
        assert dig_tdd.get("applied") is False
        assert "return 0" in (repo_tdd / "calc.py").read_text(encoding="utf-8")
    finally:
        jm_tdd.shutdown()

    # 2. Scoped sub-job with allowed_paths frames raw_user_prompt as scope-restricted reference context
    repo_scope = tmp_path / "repo_scope"
    _init_repo(repo_scope)
    (repo_scope / "a.py").write_text("A = 1\n", encoding="utf-8")
    (repo_scope / "b.py").write_text("B = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_scope), check=True)
    subprocess.run(["git", "commit", "-qm", "init scope"], cwd=str(repo_scope), check=True)

    captured_scoped: list[str] = []

    def scoped_worker(req: dict[str, Any]) -> dict[str, Any]:
        captured_scoped.extend(req["prompts"])
        (Path(req["workspace_dir"]) / "a.py").write_text("A = 2\n", encoding="utf-8")
        return {"text": "updated a.py", "usage": {"prompt_token_count": 100, "total_token_count": 110}}

    st_scope = ReceiptStore(tmp_path / "data_scope", workspace_root=repo_scope)
    (st_scope.data_dir / ".last_user_prompt.json").write_text(
        json.dumps(
            {
                "ts": time.time() + 1.0,
                "prompt": "Set A to 2 in a.py, rename B to BETA in b.py across callers, and spawn two parallel workers.",
                "session_id": "",
            }
        ),
        encoding="utf-8",
    )
    jm_scope = JobManager(workspace_root=repo_scope, store=st_scope, worker_backend=scoped_worker)
    try:
        jm_scope.delegate_start(
            "Set A = 2 in a.py only.",
            project="p",
            allowed_paths=["a.py"],
            auto_apply=False,
            wait=True,
        )
        assert "## Original User Request (Verbatim" not in captured_scoped[0]
        assert "## Background User Context (Reference Only: Scope Restricted to allowed_paths)" in captured_scoped[0]
        assert "allowed_paths=['a.py']" in captured_scoped[0]
    finally:
        jm_scope.shutdown()

    # 3. All 5 AST detector edge cases (guarded->unconditional, inherited cross-file & same-file, setattr, match/case, benign copy)
    repo_edges = tmp_path / "repo_edges"
    _init_repo(repo_edges)
    (repo_edges / "pkg").mkdir()
    (repo_edges / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo_edges / "pkg" / "base.py").write_text(
        "class RemoteBase:\n    remote_dir = '/etc/remote'\n",
        encoding="utf-8",
    )
    (repo_edges / "pkg" / "impl.py").write_text(
        "from .base import RemoteBase\n\n"
        "class LocalBase:\n"
        "    local_dir = '/etc/local'\n\n"
        "class GuardedOnHead:\n"
        "    timeout = 10\n"
        "    def __init__(self, timeout=None):\n"
        "        if timeout is not None:\n"
        "            self.timeout = timeout\n\n"
        "class ChildSameFile(LocalBase):\n"
        "    def __init__(self):\n"
        "        pass\n\n"
        "class ChildCrossFile(RemoteBase):\n"
        "    def __init__(self):\n"
        "        pass\n\n"
        "class SetattrCase:\n"
        "    path = '/a'\n"
        "    def __init__(self):\n"
        "        pass\n\n"
        "class MatchCase:\n"
        "    mode = 'a'\n"
        "    def __init__(self, k=0):\n"
        "        pass\n\n"
        "class BenignCopy:\n"
        "    items = ('a',)\n"
        "    def __init__(self):\n"
        "        pass\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_edges), check=True)
    subprocess.run(["git", "commit", "-qm", "init ast edges"], cwd=str(repo_edges), check=True)

    (repo_edges / "pkg" / "impl.py").write_text(
        "from .base import RemoteBase\n\n"
        "class LocalBase:\n"
        "    local_dir = '/etc/local'\n\n"
        "class GuardedOnHead:\n"
        "    timeout = 10\n"
        "    def __init__(self, timeout=None):\n"
        "        self.timeout = timeout or 30\n\n"
        "class ChildSameFile(LocalBase):\n"
        "    def __init__(self):\n"
        "        self.local_dir = '/tmp/x'\n\n"
        "class ChildCrossFile(RemoteBase):\n"
        "    def __init__(self):\n"
        "        self.remote_dir = '/tmp/y'\n\n"
        "class SetattrCase:\n"
        "    path = '/a'\n"
        "    def __init__(self):\n"
        "        setattr(self, 'path', '/b')\n\n"
        "class MatchCase:\n"
        "    mode = 'a'\n"
        "    def __init__(self, k=0):\n"
        "        match k:\n"
        "            case 1:\n"
        "                self.mode = 'b'\n\n"
        "class BenignCopy:\n"
        "    items = ('a',)\n"
        "    def __init__(self):\n"
        "        self.items = list(self.items)\n",
        encoding="utf-8",
    )
    edge_warnings = detect_class_attribute_shadowing(repo_edges, ["pkg/impl.py"])
    assert len(edge_warnings) == 5, f"Expected 5 warnings, got {edge_warnings}"
    assert any("GuardedOnHead.__init__ newly assigns self.timeout" in w for w in edge_warnings)
    assert any("ChildSameFile.__init__ newly assigns self.local_dir" in w for w in edge_warnings)
    assert any("ChildCrossFile.__init__ newly assigns self.remote_dir" in w for w in edge_warnings)
    assert any("SetattrCase.__init__ newly assigns self.path" in w for w in edge_warnings)
    assert any("MatchCase.__init__ newly assigns self.mode" in w for w in edge_warnings)
    assert not any("BenignCopy" in w for w in edge_warnings)


def test_opus_round2_regressions_src_layout_dotted_base_reexport_and_guard_rhs(
    tmp_path: Path,
) -> None:
    """Cover all Round 2 findings from Claude Opus 5.5 (High): above-ordering, RHS guard param check, src/dotted/reexport inheritance, and gitignored probe restoration."""
    from server.jobs import probe_baseline_acceptance, render_worker_prompt

    # 1. Scoped background header references the Lead's brief ABOVE (not below)
    rendered = render_worker_prompt(
        "BRIEF: set A = 2 in a.py",
        plan=[],
        acceptance=[],
        max_tool_calls=None,
        max_total_tokens=None,
        raw_user_prompt="USER: update a.py and b.py",
        allowed_paths=["a.py"],
        progress_channel=False,
    )
    assert rendered.index("BRIEF:") < rendered.index("## Background User Context")
    assert "implement ONLY the Lead's brief above" in rendered

    # 2. Unrelated hardcoded attribute written under `if logger is not None:` MUST still be flagged
    repo_unrel = tmp_path / "repo_unrel"
    _init_repo(repo_unrel)
    (repo_unrel / "m.py").write_text(
        "class C:\n"
        "    path = '/a'\n"
        "    def __init__(self, logger=None):\n"
        "        if logger is not None:\n"
        "            self.logger = logger\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_unrel), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo_unrel), check=True)
    (repo_unrel / "m.py").write_text(
        "class C:\n"
        "    path = '/a'\n"
        "    def __init__(self, logger=None):\n"
        "        if logger is not None:\n"
        "            self.logger = logger\n"
        "            self.path = '/hardcoded'\n",
        encoding="utf-8",
    )
    assert len(detect_class_attribute_shadowing(repo_unrel, ["m.py"])) == 1

    # 3. src/ layout absolute import, dotted `pkg.base.Base`, and `__init__.py` re-export
    repo_src = tmp_path / "repo_src"
    _init_repo(repo_src)
    (repo_src / "src" / "pkg").mkdir(parents=True)
    (repo_src / "src" / "pkg" / "__init__.py").write_text("from .base import Base\n", encoding="utf-8")
    (repo_src / "src" / "pkg" / "base.py").write_text("class Base:\n    path = '/etc/default'\n", encoding="utf-8")
    (repo_src / "src" / "pkg" / "c1.py").write_text(
        "from pkg.base import Base\nclass C1(Base):\n    def __init__(self):\n        pass\n",
        encoding="utf-8",
    )
    (repo_src / "src" / "pkg" / "c2.py").write_text(
        "import pkg.base\nclass C2(pkg.base.Base):\n    def __init__(self):\n        pass\n",
        encoding="utf-8",
    )
    (repo_src / "src" / "pkg" / "c3.py").write_text(
        "from pkg import Base\nclass C3(Base):\n    def __init__(self):\n        pass\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=str(repo_src), check=True)
    subprocess.run(["git", "commit", "-qm", "init src"], cwd=str(repo_src), check=True)

    (repo_src / "src" / "pkg" / "c1.py").write_text(
        "from pkg.base import Base\nclass C1(Base):\n    def __init__(self):\n        self.path = '/tmp/1'\n",
        encoding="utf-8",
    )
    (repo_src / "src" / "pkg" / "c2.py").write_text(
        "import pkg.base\nclass C2(pkg.base.Base):\n    def __init__(self):\n        self.path = '/tmp/2'\n",
        encoding="utf-8",
    )
    (repo_src / "src" / "pkg" / "c3.py").write_text(
        "from pkg import Base\nclass C3(Base):\n    def __init__(self):\n        self.path = '/tmp/3'\n",
        encoding="utf-8",
    )
    w_src = detect_class_attribute_shadowing(repo_src, ["src/pkg/c1.py", "src/pkg/c2.py", "src/pkg/c3.py"])
    assert len(w_src) == 3

    # 4. Pre-existing small gitignored file mutated during probe is restored cleanly
    repo_hyg = tmp_path / "repo_hyg"
    _init_repo(repo_hyg)
    (repo_hyg / ".gitignore").write_text("build/\n", encoding="utf-8")
    (repo_hyg / "t.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(repo_hyg), check=True)
    subprocess.run(["git", "commit", "-qm", "init hyg"], cwd=str(repo_hyg), check=True)
    (repo_hyg / "build").mkdir()
    (repo_hyg / "build" / "pre.bin").write_bytes(b"pre")
    probe_baseline_acceptance(
        ["echo mutated > build/pre.bin && echo new > build/new.bin && exit 1"],
        repo_hyg,
        per_cmd_timeout=5.0,
        total_timeout=8.0,
    )
    assert (repo_hyg / "build" / "pre.bin").read_bytes() == b"pre"
    assert not (repo_hyg / "build" / "new.bin").exists()


