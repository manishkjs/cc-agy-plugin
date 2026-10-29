"""Single source of truth for classifying worker verification results.

Separates verification into three explicit policies so reporting, auto-apply,
and hook guardrails never drift on compile smoke checks or discovered tests:

1. Reporting (`classify_verification`, `has_real_acceptance_pass`):
   Distinguishes `failed`, `acceptance_tests` (explicit check passed),
   `related_tests` (plugin-discovered tests passed >0 test cases), `smoke_only`
   (parse/compile/import checks only), and `unverified`.
2. Applying & timeout promotion (`satisfies_auto_apply_verification`,
   `satisfies_timeout_promotion`):
   Auto-apply compaction accepts either an explicit acceptance pass or a real
   discovered test suite (`counts.passed > 0`), never `compile_smoke` or
   `auto_import_smoke`. Timeout promotion requires every executed check to pass
   and at least one non-empty explicit acceptance check to pass.
3. Hook decisions (`is_verified_applied_receipt`, `satisfies_post_tool_auto_stop`):
   `PreToolUse` suppresses extra Lead review reads once an `ok` applied receipt
   passes real behavioral tests cleanly; `PostToolUse` auto-stops only when an
   explicit acceptance command passed.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

_NO_TESTS_OUTPUT_RE = re.compile(
    r"\bno tests ran\b|\bcollected 0 items\b|\bRan 0 tests\b",
    re.IGNORECASE,
)


def _passed_test_count(entry: Mapping[str, Any]) -> int:
    counts = entry.get("counts")
    if not isinstance(counts, Mapping):
        return 0
    try:
        return int(counts.get("passed") or 0)
    except (TypeError, ValueError):
        return 0


def _is_non_skipped_passing_entry(entry: Any) -> bool:
    return (
        isinstance(entry, Mapping)
        and entry.get("passed") is True
        and not entry.get("timed_out")
        and not entry.get("vacuous")
        and not entry.get("no_tests_collected")
        and not entry.get("runner_incompatible_skipped")
    )


def is_non_smoke_passing_entry(entry: Any) -> bool:
    """True when `entry` passed cleanly without being skipped, vacuous, or a compile/import smoke check."""
    return (
        _is_non_skipped_passing_entry(entry)
        and not entry.get("compile_smoke")
        and not entry.get("auto_import_smoke")
    )


def _is_explicit_test_entry(entry: Mapping[str, Any]) -> bool:
    return (
        is_non_smoke_passing_entry(entry)
        and not entry.get("auto_discovered_default")
        and not entry.get("auto_discovered_regression")
    )


def _is_behavioral_test_entry(entry: Mapping[str, Any]) -> bool:
    """True for an explicit acceptance pass or a discovered test suite that ran >=1 passing test."""
    if not is_non_smoke_passing_entry(entry) or entry.get("auto_discovered_regression"):
        return False
    if entry.get("auto_discovered_default"):
        return _passed_test_count(entry) > 0
    return True


@dataclass(frozen=True)
class VerificationClassification:
    tier: str
    explicit_acceptance_passed: bool
    behavioral_tests_passed: bool


def evaluate_verification(
    acceptance_results: Any,
    acceptance_passed: bool | None,
) -> VerificationClassification:
    """Compute the canonical verification tier and policy flags for a result list."""
    results = [r for r in acceptance_results if isinstance(r, Mapping)] if isinstance(acceptance_results, list) else []
    explicit_ok = any(_is_explicit_test_entry(r) for r in results)
    behavioral_ok = any(_is_behavioral_test_entry(r) for r in results)

    if acceptance_passed is False:
        return VerificationClassification(
            tier="failed",
            explicit_acceptance_passed=False,
            behavioral_tests_passed=False,
        )
    if explicit_ok:
        return VerificationClassification(
            tier="acceptance_tests",
            explicit_acceptance_passed=True,
            behavioral_tests_passed=True,
        )
    ran = [r for r in results if _is_non_skipped_passing_entry(r)]
    if any(
        not r.get("compile_smoke")
        and not r.get("auto_import_smoke")
        and _passed_test_count(r) > 0
        for r in ran
    ):
        return VerificationClassification(
            tier="related_tests",
            explicit_acceptance_passed=False,
            behavioral_tests_passed=behavioral_ok,
        )
    if ran:
        return VerificationClassification(
            tier="smoke_only",
            explicit_acceptance_passed=False,
            behavioral_tests_passed=False,
        )
    return VerificationClassification(
        tier="unverified",
        explicit_acceptance_passed=False,
        behavioral_tests_passed=False,
    )


def has_real_acceptance_pass(
    acceptance_results: Any,
    *,
    reject_zero_test_output: bool = False,
) -> bool:
    """True when at least one explicit acceptance command passed (not discovered, smoke, or skipped)."""
    if not isinstance(acceptance_results, list) or not acceptance_results:
        return False
    for entry in acceptance_results:
        if not isinstance(entry, Mapping) or not _is_explicit_test_entry(entry):
            continue
        if reject_zero_test_output and _NO_TESTS_OUTPUT_RE.search(str(entry.get("output_tail") or "")):
            continue
        return True
    return False


def classify_verification(acceptance_results: Any, acceptance_passed: bool | None) -> str:
    """Summarize verification strength: `failed`, `acceptance_tests`, `related_tests`, `smoke_only`, or `unverified`."""
    return evaluate_verification(acceptance_results, acceptance_passed).tier


def satisfies_auto_apply_verification(
    acceptance_results: Any,
    acceptance_passed: bool | None,
) -> bool:
    """Policy for `_maybe_auto_apply_digest`: explicit acceptance or discovered test suite (`passed > 0`), never smoke."""
    if acceptance_passed is not True:
        return False
    return evaluate_verification(acceptance_results, acceptance_passed).behavioral_tests_passed


def satisfies_timeout_promotion(
    acceptance_results: Any,
    *,
    test_only_diff: bool = False,
) -> bool:
    """Policy for promoting a timed-out worker with clean verified edits to `status='ok'`."""
    if test_only_diff or not isinstance(acceptance_results, list) or not acceptance_results:
        return False
    if not all(isinstance(r, Mapping) and r.get("passed") is True for r in acceptance_results):
        return False
    return has_real_acceptance_pass(acceptance_results, reject_zero_test_output=True)


def is_verified_applied_receipt(rec: Mapping[str, Any]) -> bool:
    """Hook policy (`PreToolUse`): True when an applied `ok` receipt passed real behavioral tests with no warnings."""
    if (
        rec.get("status") != "ok"
        or rec.get("applied") is not True
        or rec.get("reverted")
        or rec.get("acceptance_passed") is not True
        or rec.get("vacuous_acceptance")
        or rec.get("test_only_diff")
        or rec.get("scope_violations")
        or rec.get("missing_prompt_symbols")
        or rec.get("contract_warnings")
        or rec.get("stub_mutation_warnings")
    ):
        return False
    return satisfies_auto_apply_verification(rec.get("acceptance_results"), True)


def satisfies_post_tool_auto_stop(rec: Mapping[str, Any]) -> bool:
    """Hook policy (`PostToolUse`): True when `acceptance_passed` is True via an explicit acceptance command."""
    if rec.get("acceptance_passed") is not True:
        return False
    return has_real_acceptance_pass(rec.get("acceptance_results"))
