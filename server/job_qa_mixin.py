"""Balanced-profile pre-apply QA diff review mixin for `JobManager`."""

from __future__ import annotations

from concurrent.futures import Future
import os
from pathlib import Path
import threading
import time
from typing import Any

from server.ast_regression_guards import _is_test_file_path
from server.job_git_utils import _env_int
from server.receipts import BudgetExceededError, validate_job_id
from server.regions import env_flag, resolve_cost_profile, truncate_to_token_budget
from server.verification_policy import classify_verification
from server.worker_prompt_builder import (
    QA_REVIEW_MAX_TOKENS,
    build_qa_diff_audit_system,
    is_quota_or_rate_limit_error,
    parse_qa_audit_verdict,
)


class JobQAMixin:
    """1-shot pre-apply QA diff review and automatic worker follow-up orchestration."""

    def record_qa_review_on_receipt(
        self,
        job_id: str,
        *,
        qa_model: str,
        qa_meta: dict[str, Any],
        qa_usage: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge a 1-shot QA review's usage and metadata onto `job-*.json` without creating a separate receipt."""
        safe_id = validate_job_id(job_id)
        receipt = self.store.read_receipt(safe_id)
        prior_model = str(receipt.get("model") or "gemini-flash-next")
        u = dict(receipt.get("usage") or {})
        pq = int((qa_usage or {}).get("prompt_token_count") or 0)
        cq = int((qa_usage or {}).get("cached_content_token_count") or 0)
        oq = int((qa_usage or {}).get("candidates_token_count") or 0)
        tq = int((qa_usage or {}).get("thoughts_token_count") or 0)
        tot_q = max(int((qa_usage or {}).get("total_token_count") or 0), pq + oq + tq)
        if pq > 0 or oq > 0 or tq > 0 or tot_q > 0:
            if not u.get("requests") and any(
                int(u.get(k) or 0) > 0
                for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")
            ):
                u["requests"] = [
                    {
                        "prompt_token_count": int(u.get("prompt_token_count") or 0),
                        "cached_content_token_count": int(u.get("cached_content_token_count") or 0),
                        "candidates_token_count": int(u.get("candidates_token_count") or 0),
                        "thoughts_token_count": int(u.get("thoughts_token_count") or 0),
                        "total_token_count": int(u.get("total_token_count") or 0),
                        "model": prior_model,
                    }
                ]
            elif isinstance(u.get("requests"), list):
                tagged = [
                    ({**r, "model": r.get("model") or prior_model} if isinstance(r, dict) else r)
                    for r in u["requests"]
                ]
                sp = sum(int(r.get("prompt_token_count") or 0) for r in tagged if isinstance(r, dict))
                sc = sum(int(r.get("cached_content_token_count") or 0) for r in tagged if isinstance(r, dict))
                so = sum(int(r.get("candidates_token_count") or 0) for r in tagged if isinstance(r, dict))
                st = sum(int(r.get("thoughts_token_count") or 0) for r in tagged if isinstance(r, dict))
                rp = max(0, int(u.get("prompt_token_count") or 0) - sp)
                rc = min(rp, max(0, int(u.get("cached_content_token_count") or 0) - sc))
                ro = max(0, int(u.get("candidates_token_count") or 0) - so)
                rt = max(0, int(u.get("thoughts_token_count") or 0) - st)
                if rp > 0 or ro > 0 or rt > 0:
                    tagged.append(
                        {
                            "prompt_token_count": rp,
                            "cached_content_token_count": rc,
                            "candidates_token_count": ro,
                            "thoughts_token_count": rt,
                            "total_token_count": rp + ro + rt,
                            "model": prior_model,
                        }
                    )
                u["requests"] = tagged
            else:
                u["requests"] = []
            u["requests"].append(
                {
                    "per_call": True,
                    "single_request": True,
                    "kind": "qa_review",
                    "model": qa_model,
                    "prompt_token_count": pq,
                    "cached_content_token_count": cq,
                    "candidates_token_count": oq,
                    "thoughts_token_count": tq,
                    "total_token_count": tot_q,
                }
            )
            u["prompt_token_count"] = int(u.get("prompt_token_count") or 0) + pq
            u["cached_content_token_count"] = int(u.get("cached_content_token_count") or 0) + cq
            u["candidates_token_count"] = int(u.get("candidates_token_count") or 0) + oq
            u["thoughts_token_count"] = int(u.get("thoughts_token_count") or 0) + tq
            u["total_token_count"] = int(u.get("total_token_count") or 0) + tot_q
            receipt["usage"] = u
            precomputed = receipt.get("precomputed_cost")
            if isinstance(precomputed, dict) and "total_cost_cents" in precomputed:
                # Receipts that price their own calls (batch jobs) ignore usage when costed, so the QA
                # review's spend has to be added to the precomputed total or it never shows up.
                from server.regions import estimate_cost_usd

                qa_cost = estimate_cost_usd(
                    qa_model,
                    {"prompt_token_count": pq, "cached_content_token_count": cq, "candidates_token_count": oq,
                     "thoughts_token_count": tq, "total_token_count": tot_q},
                    single_request=True,
                )
                qa_cents = float(qa_cost.get("total_cost_cents") or 0.0)
                total_cents = float(precomputed.get("total_cost_cents") or 0.0) + qa_cents
                receipt["precomputed_cost"] = {
                    **precomputed,
                    "total_cost_cents": round(total_cents, 4),
                    "total_cost_usd": round(total_cents / 100.0, 6),
                    "pricing_known": bool(precomputed.get("pricing_known", True))
                    and bool(qa_cost.get("pricing_known", True)),
                }
        receipt["qa_review"] = dict(qa_meta)
        with self._lock:
            raw_key = self._raw_api_keys.get(safe_id)
        return self.store.write_receipt(
            safe_id,
            receipt,
            scrub=bool(receipt.get("scrub_receipt", False)),
            workspace_root=str(self.workspace_root),
            api_key=raw_key,
        )

    def _maybe_run_balanced_pre_apply_qa(
        self,
        job_id: str,
        receipt: dict[str, Any],
        *,
        auto_apply: bool | None,
        in_place: bool,
        is_cancelled_now: bool,
        wait_elapsed: float = 0.0,
        timeout_seconds: float | None = None,
        wait_budget: float | None = None,
    ) -> dict[str, Any] | None:
        """Run a 1-shot pre-apply QA diff audit on un-escalated Flash passes in `balanced`."""
        qa_flag = env_flag("CC_AGY_BALANCED_QA_REVIEW")
        if qa_flag is False:
            return None
        autonomous_solo = env_flag("CC_AGY_AUTONOMOUS_SOLO") is True
        if resolve_cost_profile() != "balanced" and not autonomous_solo:
            return None
        if self.worker_backend is not None and qa_flag is not True:
            return None
        # One extra audit round on the escalated model's final diff: escalated turns otherwise ship
        # without any review even when the stronger model leaves a requirement half-done.
        prior_qa = receipt.get("qa_review") if isinstance(receipt.get("qa_review"), dict) else None
        post_escalation_round = bool(
            receipt.get("escalated_from_model")
            and not (prior_qa or {}).get("post_escalation")
            and env_flag("CC_AGY_POST_ESCALATION_QA") is True
        )
        if (
            in_place
            or is_cancelled_now
            or receipt.get("applied")
            or receipt.get("skip_balanced_qa")
            or (receipt.get("qa_review") and not post_escalation_round)
            or (receipt.get("escalated_from_model") and not post_escalation_round)
            or receipt.get("status") != "ok"
            or receipt.get("acceptance_passed") is not True
            or not receipt.get("files_changed")
        ):
            return None
        cur_model = str(receipt.get("model") or "")
        if "flash" not in cur_model.lower() and not post_escalation_round:
            return None
        qa_model = self._resolve_escalation_target_model(cur_model) or (
            cur_model if (post_escalation_round or autonomous_solo) else None
        )
        if not qa_model:
            return None
        qa_started_mono = time.monotonic()
        esc_deadline, _esc_elapsed, rem_budget = self._session_deadline_budget()
        if esc_deadline is not None and rem_budget < (180.0 if post_escalation_round else 90.0):
            return None
        diff_path = self.store.diff_path(job_id)
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
        if not diff_text.strip():
            return None

        from server.genai_ask import ask_gemini
        from server.review_bundle import build_review_evidence_dir

        wt_candidate = Path(str(receipt.get("worktree_dir") or ""))
        if not wt_candidate.is_dir():
            wt_candidate = self.store.worktree_path(job_id)
        target_wt = wt_candidate if wt_candidate.is_dir() else self.workspace_root

        raw_user_spec = str(receipt.get("raw_user_prompt") or "").strip()
        lead_brief = str((receipt.get("prompts") or [""])[0] or "").strip()
        spec_sections: list[str] = []
        if raw_user_spec and raw_user_spec != lead_brief:
            spec_sections.append(f"### Original User Specification\n{raw_user_spec}")
            if lead_brief:
                spec_sections.append(f"### Lead Delegation Brief (Summary: Original User Specification Takes Precedence)\n{lead_brief}")
        elif lead_brief or raw_user_spec:
            spec_sections.append(lead_brief or raw_user_spec)
        spec_rel_files = [str(p) for p in (receipt.get("spec_files") or []) if p]
        for rel_sp in spec_rel_files[:4]:
            cand_sp = (target_wt / rel_sp).resolve()
            if cand_sp.is_relative_to(target_wt.resolve()) and cand_sp.is_file():
                sp_txt, _, _ = truncate_to_token_budget(
                    cand_sp.read_text(encoding="utf-8", errors="replace"),
                    1500,
                )
                spec_sections.append(f"### Spec File: {rel_sp}\n{sp_txt}")
        checklist_items = [str(c) for c in (receipt.get("checklist") or []) if c]
        if checklist_items:
            spec_sections.append("### Checklist\n" + "\n".join(f"- {c}" for c in checklist_items))
        warn_items = [
            str(w)
            for w in [*(receipt.get("stub_mutation_warnings") or []), *(receipt.get("contract_warnings") or [])]
            if w
        ]
        if receipt.get("recovered_from_checkpoint"):
            warn_items.append(
                "Worker hit MAX_TOOL_CALLS_EXCEEDED right as acceptance tests passed; verify that every "
                "requirement, edge case, and caller in the Original User Specification is completely implemented."
            )
        changed_rel = [str(p) for p in (receipt.get("files_changed") or []) if p]
        if changed_rel and not any(_is_test_file_path(p) for p in changed_rel):
            warn_items.append(
                "The diff changes production code but adds or updates no test file, so the passing suite may "
                "never exercise the requested behavior. Name the existing test that covers each requirement, "
                "or report the uncovered requirement as an ISSUE."
            )
        # Same-model self-QA only runs on warnings. A different (stronger) reviewer always gets one look:
        # Flash often writes its own passing test file, which clears every warning while the change is
        # still wrong, so a warning-gated cross-model review never fires on exactly those diffs.
        if autonomous_solo and not warn_items and qa_model == cur_model:
            return None
        if warn_items:
            spec_sections.append("### Automated Diff Warnings\n" + "\n".join(f"- {w}" for w in warn_items))
        full_spec_text = "\n\n".join(spec_sections).strip()

        acc_results = [r for r in (receipt.get("acceptance_results") or []) if isinstance(r, dict)]
        test_out_lines = [
            f"$ {r.get('command', '')} (exit {r.get('exit_code', 0)})\n{r.get('output_tail', '')}".strip()
            for r in acc_results
        ]
        test_out_str = "\n\n".join(test_out_lines)
        review_bundle = build_review_evidence_dir(
            target_wt,
            task_prompt=full_spec_text,
            diff_text=diff_text,
            test_output=test_out_str,
            files_changed=list(receipt.get("files_changed") or []),
        )
        qa_diff, qa_truncated, _qa_orig = truncate_to_token_budget(diff_text, QA_REVIEW_MAX_TOKENS)
        slices_text = str(review_bundle.get("enclosing_slices") or "")
        slices_tok = int(review_bundle.get("enclosing_slices_tokens") or 0)
        slices_block = (
            f"\n\n## Enclosing Modified Functions & Classes\n{slices_text}"
            if (slices_text and not qa_truncated and (_qa_orig + slices_tok) <= QA_REVIEW_MAX_TOKENS)
            else ""
        )
        acc_block, _, _ = truncate_to_token_budget(test_out_str, 1500) if test_out_str else ("", False, 0)
        acc_section = f"\n\n## Acceptance Command Output\n```\n{acc_block}\n```" if acc_block else ""
        qa_prompt = (
            f"## Task Specification\n{full_spec_text}\n\n"
            f"## Candidate Git Diff{' (truncated)' if qa_truncated else ''}\n```diff\n{qa_diff}\n```"
            f"{acc_section}{slices_block}"
        )
        qa_system = build_qa_diff_audit_system(
            verification=classify_verification(acc_results, receipt.get("acceptance_passed")),
            prompt=full_spec_text,
        )
        with self._lock:
            raw_proj = self._raw_projects.get(job_id)
            raw_key = self._raw_api_keys.get(job_id)
        eff_proj = raw_proj or (
            str(receipt.get("vertex_project"))
            if receipt.get("vertex_project") and receipt.get("vertex_project") != "[REDACTED]"
            else None
        )
        explicit_qa_env = (os.environ.get("CC_AGY_QA_THINKING_LEVEL") or "").strip().upper()
        raw_qa_thinking = (
            explicit_qa_env
            or (os.environ.get("CC_AGY_FORCE_WORKER_THINKING_LEVEL") or "").strip().upper()
        )
        if raw_qa_thinking in ("0", "OFF"):
            raw_qa_thinking = "ZERO"
        qa_model_low = str(qa_model).strip().lower()
        if qa_model_low.endswith("-zero") and not explicit_qa_env:
            qa_thinking = "ZERO"
        elif raw_qa_thinking in ("ZERO", "MINIMAL", "LOW", "MEDIUM", "HIGH"):
            qa_thinking = raw_qa_thinking
        elif qa_model_low.endswith("-zero"):
            qa_thinking = "ZERO"
        elif qa_model_low.endswith("-high"):
            qa_thinking = "HIGH"
        elif qa_model_low.endswith("-medium"):
            qa_thinking = "MEDIUM"
        else:
            qa_thinking = "LOW"
        qa_timeout_ms = max(5_000, _env_int("CC_AGY_QA_TIMEOUT_MS", 90_000))

        def _should_escalate_qa(
            p_qa: dict[str, Any],
            q_err: str | None,
            rem_sec: float,
            has_dl: bool,
        ) -> bool:
            if p_qa["passed"] is False:
                return (not has_dl) or rem_sec >= 120.0
            if (
                p_qa["passed"] is None
                and not is_quota_or_rate_limit_error(q_err or "")
                and (
                    bool(receipt.get("recovered_from_checkpoint"))
                    or max(int(receipt.get("tool_call_count") or 0), len(receipt.get("tool_calls") or [])) >= 15
                )
            ):
                return (not has_dl) or rem_sec >= 180.0
            return False

        def _invoke_qa_once() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
            err_s: str | None = None
            agg_u: dict[str, int] = {
                "prompt_token_count": 0,
                "cached_content_token_count": 0,
                "candidates_token_count": 0,
                "thoughts_token_count": 0,
                "total_token_count": 0,
            }
            res_q: dict[str, Any] = {}
            parsed_q: dict[str, Any] = {"verdict": "ERROR", "passed": None, "issues": []}
            for _qa_try in range(2):
                err_s = None
                try:
                    res_q = ask_gemini(
                        prompt=qa_prompt,
                        model=qa_model,
                        system_instruction=qa_system,
                        thinking_level=qa_thinking,
                        timeout_ms=qa_timeout_ms,
                        api_key=raw_key,
                        project=eff_proj,
                        location=receipt.get("vertex_location"),
                        workspace_root=self.workspace_root,
                        store=self.store,
                        scrub_receipt=bool(receipt.get("scrub_receipt", False)),
                        backend=self.ask_backend,
                        record_receipt=False,
                        budget_job_id=job_id,
                    )
                    cur_u = dict(res_q.get("usage") or {})
                    for _uk in agg_u:
                        agg_u[_uk] += int(cur_u.get(_uk) or 0)
                    parsed_q = parse_qa_audit_verdict(str(res_q.get("text") or ""))
                    break
                except Exception as qa_exc:
                    err_s = f"{type(qa_exc).__name__}: {qa_exc}"
                    res_q = {}
                    partial_u = getattr(qa_exc, "partial_usage", None)
                    if isinstance(partial_u, dict):
                        for _uk in agg_u:
                            agg_u[_uk] += int(partial_u.get(_uk) or 0)
                    parsed_q = {"verdict": "ERROR", "passed": None, "issues": []}
                    with self._lock:
                        cancelled_retry = (job_id in self._cancelled) or self._shutting_down
                    _dl_r, _, _rem_r = self._session_deadline_budget()
                    if (
                        _qa_try == 0
                        and not cancelled_retry
                        and not is_quota_or_rate_limit_error(err_s)
                        and (_dl_r is None or _rem_r >= 150.0)
                    ):
                        continue
                    break
            u_dict: dict[str, Any] = dict(agg_u) if any(agg_u.values()) else {}
            return res_q, u_dict, parsed_q, err_s

        def _build_qa_followup(parsed_q: dict[str, Any]) -> str:
            if parsed_q["issues"]:
                issues_block = "\n".join(parsed_q["issues"])
                return (
                    f"A pre-apply QA diff audit ({qa_model}) identified the following blockers against the task specification:\n"
                    f"{issues_block}\n\n"
                    f"{review_bundle['pointer_prompt']}\n\n"
                    "Fix each issue above cleanly in production code (including imported helper/caller modules) "
                    "without weakening test inputs or altering pre-existing test expectations (unless explicitly "
                    "required by the specification), and verify all tests pass."
                )
            if parsed_q.get("verdict") == "ERROR":
                return (
                    f"A pre-apply QA diff audit ({qa_model}) timed out or errored before verifying Flash's candidate patch.\n\n"
                    f"{review_bundle['pointer_prompt']}\n\n"
                    "Audit the candidate diff against every requirement and edge case in the Original User Request, "
                    "fix any missing branches or bugs in production code without weakening pre-existing test "
                    "expectations (unless explicitly required by the specification), and ensure all tests pass."
                )
            return (
                f"A pre-apply QA diff audit ({qa_model}) returned VERDICT: NEEDS_FIXES against the task specification.\n\n"
                f"{review_bundle['pointer_prompt']}\n\n"
                "Verify every requirement and edge case in the original task specification, fix any root causes in "
                "production code without weakening pre-existing test expectations (unless explicitly required by "
                "the specification), and ensure all tests pass."
            )

        slice_target = float(wait_budget) if wait_budget is not None else float(timeout_seconds or receipt.get("timeout_seconds") or 600.0)
        full_wait_cap = max(0.5, slice_target + 30.0)
        rem_qa_slice = self._resolve_wait_slice_seconds(
            slice_target,
            elapsed_seconds=wait_elapsed,
            allow_retry_grace=False,
        )
        saved_turn1_status = str(receipt.get("status") or "ok")

        if rem_qa_slice < full_wait_cap:
            qa_fut: Future[tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None]] = Future()

            def _qa_runner() -> None:
                if not qa_fut.set_running_or_notify_cancel():
                    return
                qa_fut.set_result(_invoke_qa_once())

            qa_thread = threading.Thread(target=_qa_runner, name=f"cc-agy-qa-{job_id}", daemon=True)
            with self._lock:
                self._qa_bg_threads = [t for t in self._qa_bg_threads if t.is_alive()] + [qa_thread]
            qa_thread.start()
            self._wait_for_future_or_cancel(job_id, qa_fut, timeout=rem_qa_slice)
            if not qa_fut.done():
                chain_fut: Future[dict[str, Any]] = Future()
                chain_fut.set_running_or_notify_cancel()
                running_rec = self.store.read_receipt(job_id)
                running_rec["status"] = "running"
                with self._lock:
                    raw_k_slice = self._raw_api_keys.get(job_id)
                    self._futures[job_id] = chain_fut
                    self._wait_sliced_auto_apply[job_id] = auto_apply
                running_saved = self.store.write_receipt(
                    job_id,
                    running_rec,
                    scrub=bool(running_rec.get("scrub_receipt", False)),
                    workspace_root=str(self.workspace_root),
                    api_key=raw_k_slice,
                )

                def _finish_sliced_qa() -> None:
                    try:
                        q_res, q_usage, p_qa, q_err = qa_fut.result()
                        with self._lock:
                            cancelled_bg = (job_id in self._cancelled) or self._shutting_down
                        cur_rec = self.store.read_receipt(job_id)
                        cur_rec["status"] = "cancelled" if cancelled_bg else saved_turn1_status
                        with self._lock:
                            rk = self._raw_api_keys.get(job_id)
                        self.store.write_receipt(
                            job_id,
                            cur_rec,
                            scrub=bool(cur_rec.get("scrub_receipt", False)),
                            workspace_root=str(self.workspace_root),
                            api_key=rk,
                        )
                        esc_dl_bg, _, rem_bg = self._session_deadline_budget()
                        do_esc = (
                            not cancelled_bg
                            and _should_escalate_qa(p_qa, q_err, rem_bg, esc_dl_bg is not None)
                        )
                        q_meta: dict[str, Any] = {
                            "model": qa_model,
                            "job_id": q_res.get("job_id"),
                            "verdict": p_qa["verdict"],
                            "passed": p_qa["passed"],
                            "issues": p_qa["issues"],
                            "resumed": bool(do_esc),
                            **({"post_escalation": True} if post_escalation_round else {}),
                            "review_dir": review_bundle["review_dir"],
                            "pointer_prompt_tokens": review_bundle["pointer_prompt_tokens"],
                            "usage": q_usage or None,
                            "cost": q_res.get("cost"),
                        }
                        if q_err:
                            q_meta["error"] = q_err
                        merged_rec = self.record_qa_review_on_receipt(
                            job_id,
                            qa_model=qa_model,
                            qa_meta=q_meta,
                            qa_usage=q_usage,
                        )
                        if cancelled_bg:
                            if not in_place and target_wt != self.workspace_root and target_wt.exists():
                                self._cleanup_worktree(target_wt)
                            chain_fut.set_result(merged_rec)
                            return
                        if not do_esc:
                            chain_fut.set_result(merged_rec)
                            return
                        wt_to = float(timeout_seconds or receipt.get("timeout_seconds") or 600.0)
                        if esc_dl_bg is not None:
                            wt_to = min(wt_to, max(60.0, rem_bg))
                        with self._lock:
                            if self._futures.get(job_id) is chain_fut:
                                self._futures.pop(job_id, None)
                        try:
                            cont_bg = self.delegate_continue(
                                job_id=job_id,
                                follow_up_prompt=_build_qa_followup(p_qa),
                                allowed_paths=receipt.get("allowed_paths"),
                                acceptance=None,
                                timeout_seconds=wt_to,
                                wait=False,
                                scrub_receipt=bool(receipt.get("scrub_receipt", False)),
                                auto_apply=auto_apply,
                                _prompt_source="plugin_qa_review",
                            )
                        except Exception:
                            q_meta["resumed"] = False
                            fallback_rec = self.record_qa_review_on_receipt(
                                job_id,
                                qa_model=qa_model,
                                qa_meta=q_meta,
                                qa_usage=None,
                            )
                            chain_fut.set_result(fallback_rec)
                            return
                        if cont_bg.get("continued") is False:
                            q_meta["resumed"] = False
                            fallback_rec = self.record_qa_review_on_receipt(
                                job_id,
                                qa_model=qa_model,
                                qa_meta=q_meta,
                                qa_usage=None,
                            )
                            chain_fut.set_result(fallback_rec)
                            return
                        with self._lock:
                            self._wait_sliced_auto_apply[job_id] = auto_apply
                            next_fut = self._futures.get(job_id)
                        if next_fut is not None and next_fut is not chain_fut:
                            try:
                                chain_fut.set_result(next_fut.result())
                            except Exception as nf_exc:
                                chain_fut.set_exception(nf_exc)
                        else:
                            chain_fut.set_result(self.store.read_receipt(job_id))
                    except Exception as bg_exc:
                        if not chain_fut.done():
                            chain_fut.set_exception(bg_exc)

                chain_thread = threading.Thread(
                    target=_finish_sliced_qa,
                    name=f"cc-agy-qa-chain-{job_id}",
                    daemon=True,
                )
                with self._lock:
                    self._qa_bg_threads = [t for t in self._qa_bg_threads if t.is_alive()] + [chain_thread]
                chain_thread.start()
                return self._digest_with_progress(running_saved)
            qa_res, qa_usage, parsed_qa, qa_error = qa_fut.result()
        else:
            qa_res, qa_usage, parsed_qa, qa_error = _invoke_qa_once()

        will_escalate = _should_escalate_qa(parsed_qa, qa_error, rem_budget, esc_deadline is not None)
        qa_meta: dict[str, Any] = {
            "model": qa_model,
            "job_id": qa_res.get("job_id"),
            "verdict": parsed_qa["verdict"],
            "passed": parsed_qa["passed"],
            "issues": parsed_qa["issues"],
            "resumed": bool(will_escalate),
            **({"post_escalation": True} if post_escalation_round else {}),
            "review_dir": review_bundle["review_dir"],
            "pointer_prompt_tokens": review_bundle["pointer_prompt_tokens"],
            "usage": qa_usage or None,
            "cost": qa_res.get("cost"),
        }
        if qa_error:
            qa_meta["error"] = qa_error

        self.record_qa_review_on_receipt(
            job_id,
            qa_model=qa_model,
            qa_meta=qa_meta,
            qa_usage=qa_usage,
        )
        if not will_escalate:
            return None

        followup = _build_qa_followup(parsed_qa)
        post_qa_elapsed = max(0.0, float(wait_elapsed) + max(0.0, time.monotonic() - qa_started_mono))
        esc_deadline_after_qa, _esc_elapsed_after_qa, rem_budget_after_qa = self._session_deadline_budget()
        worker_timeout = float(timeout_seconds or receipt.get("timeout_seconds") or 600.0)
        if esc_deadline_after_qa is not None:
            worker_timeout = min(worker_timeout, max(60.0, rem_budget_after_qa))
        try:
            cont_digest = self.delegate_continue(
                job_id=job_id,
                follow_up_prompt=followup,
                allowed_paths=receipt.get("allowed_paths"),
                acceptance=None,
                timeout_seconds=worker_timeout,
                wait=True,
                scrub_receipt=bool(receipt.get("scrub_receipt", False)),
                auto_apply=auto_apply,
                _wait_elapsed=post_qa_elapsed,
                _wait_budget=wait_budget,
                _prompt_source="plugin_qa_review",
            )
            if cont_digest.get("continued") is False:
                qa_meta["resumed"] = False
                self.record_qa_review_on_receipt(
                    job_id,
                    qa_model=qa_model,
                    qa_meta=qa_meta,
                    qa_usage=None,
                )
                cont_digest["qa_review"] = dict(qa_meta)
            return cont_digest
        except BudgetExceededError as exc:
            qa_meta["resumed"] = False
            updated_rec = self.record_qa_review_on_receipt(
                job_id,
                qa_model=qa_model,
                qa_meta=qa_meta,
                qa_usage=None,
            )
            digest = self._digest_with_progress(updated_rec)
            digest["budget_exceeded"] = True
            digest["budget_exceeded_note"] = str(exc)
            return digest
