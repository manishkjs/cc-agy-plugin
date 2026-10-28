"""Model Context Protocol (MCP) JSON-RPC 2.0 server for cc-agy-plugin."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import math
import os
import sys
import threading
from typing import Any, BinaryIO

from server.doctor import run_doctor
from server.genai_ask import ask_gemini
from server.jobs import (
    QA_DIFF_AUDIT_SYSTEM,
    QA_REVIEW_MAX_TOKENS,
    JobManager,
    build_qa_diff_audit_system,
    parse_qa_audit_verdict,
    validate_acceptance,
    validate_allowed_paths,
    validate_plan,
)
from server.receipts import BudgetExceededError, ReceiptStore
from server.regions import env_flag, is_strict_mode_enabled, truncate_to_token_budget
from server.true_cost import true_total_report
from server.batch_plan import BatchPlanError, load_batch_plan, validate_batch_plan
from server.batch_runner import FileBatchRunner, batch_task_log, probe_writers
from server.assist_tools import AssistRequestError, prepare_document, prepare_replication, prepare_tests
from server.model_routes import resolve_batch_settings
from server.regions import resolve_auth_config
from server.review_bundle import build_review_evidence_dir

MCP_PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_MCP_VERSIONS = frozenset({"2024-11-05", "2025-03-26"})

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "ask_gemini",
        "description": (
            "Run a fast single-shot Vertex AI or AI Studio Gemini query via google-genai "
            "(without loading the Antigravity SDK's built-in agent system prompt and tool "
            "declarations). Best for code reviews, architecture feedback, and second opinions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Question or review request for Gemini.",
                },
                "model": {
                    "type": "string",
                    "description": "Vertex Gemini model ID (default: gemini-3.8-flash; respects cost_profile).",
                },
                "context_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of workspace-relative file paths to attach.",
                },
                "system_instruction": {
                    "type": "string",
                    "description": "Optional system instruction.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Optional sampling temperature.",
                },
                "thinking_level": {
                    "type": "string",
                    "enum": ["MINIMAL", "LOW", "MEDIUM", "HIGH", "EXTRA_HIGH"],
                    "description": "Optional reasoning depth for thinking-capable Gemini models.",
                },
                "api_key": {
                    "type": "string",
                    "description": "Optional Google AI Studio Gemini API key. Overrides GEMINI_API_KEY environment variable and bypasses Vertex AI requirements.",
                },
                "project": {
                    "type": "string",
                    "description": "Optional Google Cloud project ID override for Vertex AI.",
                },
                "location": {
                    "type": "string",
                    "description": "Optional Vertex AI region override (auto-routed to global when required).",
                },
                "scrub_receipt": {
                    "type": "boolean",
                    "description": "If true, redact paying project ID and host workspace paths in the saved receipt.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "delegate_start",
        "description": (
            "Start an isolated Antigravity SDK worker in a dedicated standalone repo copy. "
            "Returns a job_id immediately (or waits if wait=true). Failed runs never "
            "pollute the main working tree."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Concise, outcome-oriented task brief (keep under 200 tokens when passing spec_files/context_files, "
                        "or under 500 words otherwise; state the behavioral requirements, edge cases, and target file paths "
                        "directly without 'Wait:' self-debate or speculative monologues; avoid pasting >8 consecutive lines "
                        "of implementation code; the worker receives a cached line-numbered Repository Map and Focused File Outline)."
                    ),
                },
                "spec_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional workspace-relative paths to specification or instruction files (e.g. ['PROMPT.md', 'docs/spec.md']) "
                        "whose contents are inlined verbatim into the worker's Turn-1 prompt so you do not have to re-type them."
                    ),
                },
                "context_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional workspace-relative paths to target/source files whose contents are inlined into the worker's "
                        "Turn-1 prompt so the worker can edit immediately without calling view_file."
                    ),
                },
                "checklist": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of concise behavioral requirement items for the worker to implement and verify.",
                },
                "model": {
                    "type": "string",
                    "description": "Vertex Gemini model ID (default: gemini-3.8-flash, or gemini-4-argon when cost_profile='quality').",
                },
                "allowed_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of workspace-relative file paths or directory prefixes this worker may modify.",
                },
                "enabled_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional allowlist of Antigravity worker tools.",
                },
                "disabled_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional denylist of Antigravity worker tools.",
                },
                "enable_subagents": {
                    "type": "boolean",
                    "description": "Enable nested subagents inside the Antigravity worker (default: false).",
                },
                "plan": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered plan steps for the worker (outcomes, not code). The worker reports each step DONE or BLOCKED.",
                },
                "acceptance": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Up to 5 targeted, read-only check commands (prefer focused test files such as "
                        "`pytest -q tests/test_target.py` rather than slow full-repo suites). The plugin runs "
                        "them after the worker finishes and returns pass/fail with output tails and diff_preview, "
                        "so you can call delegate_apply directly without re-running get_diff or Bash tests."
                    ),
                },
                "max_tool_calls": {
                    "type": "integer",
                    "description": "Per-turn worker cap (default 100, env CC_AGY_MAX_TOOL_CALLS; 0 = no cap). When hit, status becomes 'checkpoint' and you decide whether to continue.",
                },
                "max_total_tokens": {
                    "type": "integer",
                    "description": "Per-turn worker token cap (default 5,000,000, env CC_AGY_MAX_TOTAL_TOKENS; 0 = no cap).",
                },
                "thinking_level": {
                    "type": "string",
                    "enum": ["MINIMAL", "LOW", "MEDIUM", "HIGH", "EXTRA_HIGH", "DEFAULT"],
                    "description": "Worker reasoning depth (default HIGH, or CC_AGY_DELEGATE_THINKING_LEVEL). DEFAULT leaves it to the model.",
                },
                "in_place": {
                    "type": "boolean",
                    "description": (
                        "Edit the real working tree directly instead of an isolated copy (default: false). "
                        "Needs a clean tracked tree; one in_place job at a time; cannot be combined with "
                        "allowed_paths. The job counts as applied when it finishes; undo_hint shows how to revert."
                    ),
                },
                "scrub_receipt": {
                    "type": "boolean",
                    "description": "Redact paying project ID and host paths in the receipt (default: false).",
                },
                "api_key": {
                    "type": "string",
                    "description": "Optional Google AI Studio Gemini API key. Overrides GEMINI_API_KEY environment variable and bypasses Vertex AI requirements.",
                },
                "project": {
                    "type": "string",
                    "description": "Optional Google Cloud project ID for Vertex AI.",
                },
                "location": {
                    "type": "string",
                    "description": "Optional Vertex AI region override.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Per-turn timeout in seconds (default: 1200).",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true, block until the worker finishes and return the digest.",
                },
                "preload_files": {
                    "type": "boolean",
                    "description": "If true, list the top keyword-matched repository file paths (paths only, not contents) in the initial worker prompt so the worker opens the right files first (default: true).",
                },
                "auto_apply": {
                    "type": "boolean",
                    "description": "If true (default in strict orchestrator mode or when CC_AGY_AUTO_APPLY_ON_PASS=1) and wait=true, automatically apply the diff when status='ok', acceptance checks pass, and scope_violations=[].",
                },
                "qa_review_model": {
                    "type": "string",
                    "description": "Optional stronger Vertex Gemini model ID (or env CC_AGY_QA_REVIEW_MODEL) to run an automated QA diff audit when wait=true.",
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "delegate_status",
        "description": (
            "Check status, token usage, and estimated cost for a specific job_id, "
            "or omit job_id to list recent jobs and cumulative session costs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Optional job_id to inspect.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of recent jobs to return in list mode (default: 20, max: 50).",
                },
            },
        },
    },
    {
        "name": "delegate_result",
        "description": "Wait for a delegated job to finish and return its token-bounded digest.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID returned by delegate_start.",
                },
                "wait": {
                    "type": "boolean",
                    "description": "Whether to wait if the job is still running (default: true).",
                },
                "timeout": {
                    "type": "number",
                    "description": "Max seconds to wait (default: 240, so your prompt cache stays warm between polls; max 540).",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "get_diff",
        "description": (
            "Retrieve the unified git diff from a worker job's isolated worktree, "
            "paginated by token budget so responses never exceed Claude Code's MCP token cap."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID to inspect.",
                },
                "file_filter": {
                    "type": "string",
                    "description": "Optional exact relative path, directory prefix, basename, or glob filter.",
                },
                "page": {
                    "type": "integer",
                    "description": "1-based page number (default: 1).",
                },
                "page_tokens": {
                    "type": "integer",
                    "description": "Token budget per page (default: 4000, max: 8000).",
                },
                "summary_only": {
                    "type": "boolean",
                    "description": "If true, return only per-file added/removed line stats (`file_stats`) and token estimates without diff hunks (default: false).",
                },
                "focus_symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of symbol names; filters diff output to only hunks touching at least one symbol.",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "delegate_apply",
        "description": (
            "Apply a verified worker job's changes from its isolated worktree to the main "
            "workspace tree, record provenance hashes on the receipt, and clean up the worktree."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Completed job_id.",
                },
                "cleanup_worktree": {
                    "type": "boolean",
                    "description": "Remove the temporary worktree after applying (default: true).",
                },
                "allow_scope_violations": {
                    "type": "boolean",
                    "description": (
                        "Allow applying files outside allowed_paths if the Lead has reviewed the diff "
                        "with get_diff and approves the additional modified files (default: false)."
                    ),
                },
                "include_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional path/glob patterns for untracked out-of-scope files to explicitly include "
                        "when allow_scope_violations=true."
                    ),
                },
                "include_unverified_new_tests": {
                    "type": "boolean",
                    "description": (
                        "Include newly created untracked test files even if acceptance_passed is not true "
                        "(default: false)."
                    ),
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "delegate_continue",
        "description": "Send a follow-up instruction to an existing worker job's worktree.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Existing job_id.",
                },
                "follow_up_prompt": {
                    "type": "string",
                    "description": "Follow-up task or test-fix instruction.",
                },
                "allowed_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional updated list of allowed relative paths for this job "
                        "(pass [] to remove path restrictions, or omit to keep existing allowed_paths)."
                    ),
                },
                "acceptance": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional updated list of read-only acceptance check commands.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 600).",
                },
                "wait": {
                    "type": "boolean",
                    "description": "Wait for the follow-up turn to finish (default: true).",
                },
                "api_key": {
                    "type": "string",
                    "description": "Optional Google AI Studio Gemini API key override.",
                },
                "project": {
                    "type": "string",
                    "description": "Optional Google Cloud project ID override for Vertex AI.",
                },
                "location": {
                    "type": "string",
                    "description": "Optional Vertex AI region override.",
                },
                "force_retry": {
                    "type": "boolean",
                    "description": (
                        "Run another Gemini attempt even though the job was escalated to the Lead after "
                        "repeated failures (default: false)."
                    ),
                },
                "auto_apply": {
                    "type": "boolean",
                    "description": "If true (or env CC_AGY_AUTO_APPLY_ON_PASS=1) and wait=true, automatically apply the diff when status='ok', acceptance checks pass, and scope_violations=[].",
                },
            },
            "required": ["job_id", "follow_up_prompt"],
        },
    },
    {
        "name": "delegate_revert",
        "description": (
            "Undo an applied job: restore every file it touched to its pre-apply state from the backups "
            "recorded at apply time. Refuses if any of those files changed after the apply, unless skip_dirty "
            "(a partial revert: call again once the skipped files match the applied content to finish it)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Applied job_id (the receipt id)."},
                "receipt_id": {"type": "string", "description": "Alias of job_id."},
                "dry_run": {"type": "boolean", "description": "Report what would be restored without writing (default: false)."},
                "skip_dirty": {
                    "type": "boolean",
                    "description": "Revert only the files that still match what the job applied (default: false).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "cost_report",
        "description": (
            "Dispatched cost (Gemini calls the plugin sent, from receipts) vs true total cost (dispatched plus "
            "the Claude Code lead session, rebuilt from its transcripts) for this workspace and time window."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "since_epoch": {"type": "number", "description": "Window start, epoch seconds (default: first job in this workspace)."},
                "until_epoch": {"type": "number", "description": "Window end, epoch seconds (default: now)."},
                "transcripts_dir": {
                    "type": "string",
                    "description": "Claude Code transcript folder (default: $CLAUDE_CONFIG_DIR or ~/.claude, projects/<workspace hash>).",
                },
            },
        },
    },
    {
        "name": "delegate_cancel",
        "description": "Cancel a running worker job and remove its temporary worktree.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Job ID to cancel.",
                },
                "cleanup_worktree": {
                    "type": "boolean",
                    "description": "Remove the temporary worktree (default: true).",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "delegate_to_gemini",
        "description": (
            "Synchronous convenience tool: runs an Antigravity worker in an isolated "
            "worktree, waits for completion, and optionally applies the verified diff if status='ok'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Task specification for the Gemini worker.",
                },
                "model": {
                    "type": "string",
                    "description": "Vertex Gemini model ID (default: gemini-3.8-flash, or gemini-4-argon when cost_profile='quality').",
                },
                "allowed_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of allowed relative paths.",
                },
                "enabled_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional allowlist of Antigravity worker tools.",
                },
                "disabled_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional denylist of Antigravity worker tools.",
                },
                "enable_subagents": {
                    "type": "boolean",
                    "description": "Enable nested subagents inside the Antigravity worker (default: false).",
                },
                "plan": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered plan steps for the worker (outcomes, not code). The worker reports each step DONE or BLOCKED.",
                },
                "acceptance": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to 5 read-only check commands (e.g. focused pytest runs). The plugin runs them after the worker finishes and returns pass/fail with output tails, so you do not need to re-run them.",
                },
                "max_tool_calls": {
                    "type": "integer",
                    "description": "Per-turn worker cap (default 100, env CC_AGY_MAX_TOOL_CALLS; 0 = no cap). When hit, status becomes 'checkpoint' and you decide whether to continue.",
                },
                "max_total_tokens": {
                    "type": "integer",
                    "description": "Per-turn worker token cap (default 5,000,000, env CC_AGY_MAX_TOTAL_TOKENS; 0 = no cap).",
                },
                "thinking_level": {
                    "type": "string",
                    "enum": ["MINIMAL", "LOW", "MEDIUM", "HIGH", "EXTRA_HIGH", "DEFAULT"],
                    "description": "Worker reasoning depth (default HIGH, or CC_AGY_DELEGATE_THINKING_LEVEL). DEFAULT leaves it to the model.",
                },
                "in_place": {
                    "type": "boolean",
                    "description": (
                        "Edit the real working tree directly instead of an isolated copy (default: false). "
                        "Needs a clean tracked tree; one in_place job at a time; cannot be combined with "
                        "allowed_paths. The job counts as applied when it finishes; undo_hint shows how to revert."
                    ),
                },
                "scrub_receipt": {
                    "type": "boolean",
                    "description": "Redact paying project ID and host paths in the receipt (default: false).",
                },
                "api_key": {
                    "type": "string",
                    "description": "Optional Google AI Studio Gemini API key. Overrides GEMINI_API_KEY environment variable and bypasses Vertex AI requirements.",
                },
                "project": {
                    "type": "string",
                    "description": "Optional Google Cloud project ID for Vertex AI.",
                },
                "location": {
                    "type": "string",
                    "description": "Optional Vertex AI region override.",
                },
                "auto_apply": {
                    "type": "boolean",
                    "description": "Automatically apply the worktree diff to the main workspace if status='ok' (default: true).",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 1200).",
                },
                "preload_files": {
                    "type": "boolean",
                    "description": "If true, list the top keyword-matched repository file paths (paths only, not contents) in the initial worker prompt so the worker opens the right files first (default: true).",
                },
                "qa_review_model": {
                    "type": "string",
                    "description": (
                        "Optional 1-shot QA reviewer model (e.g. 'gemini-4-argon'). When set, audits the worker's "
                        "git diff (capped at 25,000 tokens, HIGH thinking) against the task prompt after Turn 1 and, "
                        "if concrete bugs are found, runs delegate_continue as a fresh worker session on the edited "
                        "worktree before auto_apply. Experimental: not yet shown to raise solve rate."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
]

_AUTH_PROPS: dict[str, Any] = {
    "api_key": {"type": "string", "description": "Optional Google AI Studio Gemini API key override."},
    "project": {"type": "string", "description": "Optional Google Cloud project ID override for Vertex AI."},
    "location": {"type": "string", "description": "Optional Vertex AI region override."},
}

_WRITER_PROPS: dict[str, Any] = {
    "writer": {"type": "string", "enum": ["genai", "agent"],
               "description": "Low-cost writer: 'genai' (one generate_content call, default) or 'agent' (Antigravity, writer-only or bounded reader)."},
    "model": {"type": "string", "description": "Writer model (default gemini-3.8-flash, 'gemini-4-argon', or CC_AGY_BATCH_MODEL)."},
    "thinking_level": {"type": "string", "enum": ["MINIMAL", "LOW", "MEDIUM", "HIGH"], "description": "Writer thinking (default LOW)."},
    "read_budget": {"type": "integer", "description": "Bounded read-only lookups per task on the agent writer (default 0 = blind writer, max 12; e.g. 6 on messy brownfield repos)."},
    "claude_writer": {"type": "boolean", "description": "Use Claude (claude -p, no tools) as the third attempt (default true)."},
    "claude_model": {"type": "string", "description": "Claude model for the third attempt (default opus)."},
    "max_parallel": {"type": "integer", "description": "Writer calls in flight at once (default 4, max 16); checks run one task at a time."},
    "max_cost_cents": {"type": "number", "description": "Stop starting new work once the batch has spent this much (cents)."},
    "max_claude_escalations": {"type": "integer", "description": "Cap on Claude third attempts per batch (default 10)."},
    "check_timeout_seconds": {"type": "number", "description": "Per-check timeout, clamped to 5-3600 s (default 300)."},
}

TOOL_DEFINITIONS.extend([
    {
        "name": "delegate_file_batch",
        "description": (
            "Write many files from one plan in a single call. Each task is answered by a single-shot writer "
            "(no exploration), checked in an isolated job copy, retried with the failing output, and escalated "
            "to Claude on the third attempt. Returns a compact per-task digest; land the job with get_diff + "
            "delegate_apply. Use for 6+ file changes with a clear per-file plan."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_path": {"type": "string", "description": "Batch plan JSON written by the Lead (outside the repo, e.g. /tmp/plan.json, or inside it)."},
                "tasks": {"type": "array", "items": {"type": "object"}, "description": "Inline tasks instead of plan_path."},
                "shared_context": {"type": "array", "items": {"type": "string"}, "description": "Inline plans only: files every task sees."},
                "acceptance": {"type": "array", "items": {"type": "string"}, "description": "Read-only commands run once after all tasks."},
                "formatter": {"type": "string", "description": "Inline plans only: allowlisted formatter, e.g. 'ruff format {path}'."},
                "allowed_paths": {"type": "array", "items": {"type": "string"}, "description": "Restrict which paths tasks may write."},
                "resume_job_id": {"type": "string", "description": "Continue an unapplied batch job; identical re-sent plans resume automatically."},
                **_WRITER_PROPS,
                **_AUTH_PROPS,
            },
        },
    },
    {
        "name": "draft_document",
        "description": (
            "Draft one new doc or spec from purpose, audience, sections and facts. Each fact's quote must exist "
            "verbatim in its source file; the draft must contain every section and only link to real paths. "
            "Verified in an isolated job copy; land with delegate_apply."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "purpose": {"type": "string"},
                "audience": {"type": "string"},
                "sections": {"type": "array", "items": {"type": "string"}},
                "facts": {"type": "array", "items": {"type": "object", "properties": {
                    "claim": {"type": "string"}, "source": {"type": "string"}, "quote": {"type": "string"}}}},
                "context_files": {"type": "array", "items": {"type": "string"}},
                **_WRITER_PROPS,
                **_AUTH_PROPS,
            },
            "required": ["path", "purpose", "audience", "sections"],
        },
    },
    {
        "name": "write_tests_from_cases",
        "description": (
            "Write one new test file for a target file from named cases (name, given, expect). Every case and every "
            "expected value must appear verbatim and test_command must pass in an isolated job copy."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "test_path": {"type": "string"},
                "target_path": {"type": "string"},
                "cases": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "given": {"type": "string"}, "expect": {"type": "string"}}}},
                "test_command": {"type": "string", "description": "Read-only test command, e.g. python -m pytest -q tests/test_x.py"},
                **_WRITER_PROPS,
                **_AUTH_PROPS,
            },
            "required": ["test_path", "target_path", "cases", "test_command"],
        },
    },
    {
        "name": "replicate_edit",
        "description": (
            "Repeat a change you already made by hand in example_path (uncommitted) across target files as exact "
            "find/replace edits, then run check_command once in an isolated job copy."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "example_path": {"type": "string"},
                "targets": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string", "description": "Optional one-line guidance, e.g. what not to touch."},
                "check_command": {"type": "string"},
                **_WRITER_PROPS,
                **_AUTH_PROPS,
            },
            "required": ["example_path", "targets"],
        },
    },
    {
        "name": "batch_task_log",
        "description": "Attempt history, last failure and recent check log paths for one task of a batch job.",
        "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string"}, "task_id": {"type": "string"}},
                        "required": ["job_id", "task_id"]},
    },
    {
        "name": "probe_writers",
        "description": (
            "Preflight: send one tiny request through each configured batch writer (a fraction of a cent each). "
            "Fails fast with the exact fix when one cannot answer (auth, missing CLI); warns on 429/5xx."
        ),
        "inputSchema": {"type": "object", "properties": {**_WRITER_PROPS, **_AUTH_PROPS}},
    },
    {
        "name": "doctor",
        "description": (
            "Run offline readiness diagnostics (zero model or network calls): checks MCP tool health, "
            "cost_profile, authentication config (without exposing API keys), installed SDKs "
            "(google-genai, google.antigravity, Straitjacket ctx), git workspace status, detected "
            "language test runner, and receipt store writability."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
])

KNOWN_TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in TOOL_DEFINITIONS)

CORE_TOOL_NAMES: tuple[str, ...] = (
    "delegate_start",
    "delegate_continue",
    "delegate_apply",
    "delegate_result",
    "delegate_status",
    "delegate_cancel",
    "delegate_revert",
    "get_diff",
    "doctor",
)

_CORE_OMIT_PROPERTIES: frozenset[str] = frozenset(
    {
        "api_key",
        "project",
        "location",
        "scrub_receipt",
        "enabled_tools",
        "disabled_tools",
        "enable_subagents",
        "in_place",
        "qa_review_model",
        "preload_files",
        "max_total_tokens",
        "limit",
        "receipt_id",
        "dry_run",
        "skip_dirty",
        "include_unverified_new_tests",
    }
)

_CORE_CONCISE_TOOL_DESCRIPTIONS: dict[str, str] = {
    "delegate_start": "Run an isolated Gemini worker copy; auto-applies when wait=true and acceptance passes.",
    "delegate_result": "Wait up to 240s for a running job_id and return its bounded digest.",
    "delegate_status": "Check status/cost for job_id or list recent jobs.",
    "delegate_cancel": "Cancel a running job_id and clean up its worktree.",
    "delegate_revert": "Restore files touched by an applied job_id to their pre-apply state.",
    "get_diff": "Read paginated unified diff or file_stats for a job_id.",
    "delegate_apply": "Apply a completed job_id's changes to the workspace.",
    "delegate_continue": "Continue an existing job_id in its worktree with follow_up_prompt.",
    "doctor": "Run local readiness diagnostics (auth, SDKs, git, test runner).",
}

_CORE_CONCISE_PROP_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("delegate_start", "prompt"): "Concise task brief (<500 words; no self-debate or >8-line code blocks).",
    ("delegate_start", "spec_files"): "Spec paths inlined verbatim.",
    ("delegate_start", "context_files"): "Source paths inlined into worker prompt.",
    ("delegate_start", "checklist"): "Requirement items to verify.",
    ("delegate_start", "model"): "Gemini model ID.",
    ("delegate_start", "allowed_paths"): "Relative paths worker may edit.",
    ("delegate_start", "plan"): "Ordered outcome steps.",
    ("delegate_start", "acceptance"): "Up to 5 read-only test commands.",
    ("delegate_start", "max_tool_calls"): "Worker tool cap.",
    ("delegate_start", "thinking_level"): "Reasoning depth.",
    ("delegate_start", "timeout_seconds"): "Worker timeout seconds.",
    ("delegate_start", "wait"): "Block up to 240s slice.",
    ("delegate_start", "auto_apply"): "Auto-apply on passing acceptance.",
    ("delegate_continue", "job_id"): "Job ID.",
    ("delegate_continue", "follow_up_prompt"): "Follow-up instruction.",
    ("delegate_continue", "allowed_paths"): "Updated allowed paths.",
    ("delegate_continue", "acceptance"): "Updated acceptance commands.",
    ("delegate_continue", "timeout_seconds"): "Worker timeout seconds.",
    ("delegate_continue", "wait"): "Wait for completion.",
    ("delegate_continue", "force_retry"): "Retry past escalation.",
    ("delegate_continue", "auto_apply"): "Auto-apply on pass.",
    ("delegate_result", "job_id"): "Job ID.",
    ("delegate_result", "wait"): "Wait if running.",
    ("delegate_result", "timeout"): "Max wait seconds (default 240).",
    ("delegate_status", "job_id"): "Optional job ID.",
    ("delegate_cancel", "job_id"): "Job ID.",
    ("delegate_cancel", "cleanup_worktree"): "Remove worktree.",
    ("delegate_revert", "job_id"): "Applied job ID.",
    ("get_diff", "job_id"): "Job ID.",
    ("get_diff", "file_filter"): "Path/glob filter.",
    ("get_diff", "page"): "1-based page.",
    ("get_diff", "page_tokens"): "Tokens per page.",
    ("get_diff", "summary_only"): "Only file_stats.",
    ("get_diff", "focus_symbols"): "Filter hunks by symbol.",
    ("delegate_apply", "job_id"): "Job ID.",
    ("delegate_apply", "cleanup_worktree"): "Remove worktree after apply.",
    ("delegate_apply", "allow_scope_violations"): "Approve out-of-scope edits.",
    ("delegate_apply", "include_paths"): "Untracked out-of-scope paths to include.",
}


def _compact_core_tool_def(tool_def: dict[str, Any]) -> dict[str, Any]:
    name = str(tool_def.get("name") or "")
    raw_schema = tool_def.get("inputSchema") or {}
    raw_props = raw_schema.get("properties") or {}
    compact_props: dict[str, Any] = {}
    for prop_name, prop_val in raw_props.items():
        if prop_name in _CORE_OMIT_PROPERTIES or not isinstance(prop_val, dict):
            continue
        short_desc = _CORE_CONCISE_PROP_DESCRIPTIONS.get((name, prop_name))
        if short_desc is not None:
            compact_props[prop_name] = {**prop_val, "description": short_desc}
        else:
            compact_props[prop_name] = dict(prop_val)
    compact_schema: dict[str, Any] = {
        "type": raw_schema.get("type", "object"),
        "properties": compact_props,
    }
    if "required" in raw_schema:
        compact_schema["required"] = list(raw_schema["required"])
    return {
        "name": name,
        "description": _CORE_CONCISE_TOOL_DESCRIPTIONS.get(name, str(tool_def.get("description") or "")),
        "inputSchema": compact_schema,
    }


HOST_REVIEW_INSTRUCTIONS = (
    "Gemini worker workflow: 1) Call delegate_start(prompt, checklist=[...], acceptance=[...], wait=true) with "
    "the requirements and, when the repository has tests, the test command that proves the change. "
    "2) Read the digest: verification='acceptance_tests' means an explicit acceptance command passed; "
    "'smoke_only', 'related_tests' or 'unverified' means no explicit test covered the request (common in "
    "repositories without tests), so review diff_preview or get_diff against every requirement yourself. "
    "3) Land the change with delegate_apply, send the missing items with delegate_continue, or discard it with "
    "delegate_cancel. delegate_revert undoes an applied job."
)


def server_instructions_for_host(env: dict[str, str] | None = None) -> str | None:
    """MCP `initialize` instructions for hosts without plugin hooks (for example Codex).

    Claude Code already receives the same checklist from the SessionStart hook, so the instructions are
    omitted there to avoid paying for it twice. `CC_AGY_MCP_INSTRUCTIONS=1|0` forces them on or off.
    """
    active_env = os.environ if env is None else env
    forced = env_flag("CC_AGY_MCP_INSTRUCTIONS", active_env)
    if forced is False:
        return None
    if forced is not True:
        claude_root = str(active_env.get("CLAUDE_PLUGIN_ROOT") or "").strip()
        if (claude_root and not claude_root.startswith("${")) or str(active_env.get("CLAUDECODE") or "") == "1":
            return None
    return HOST_REVIEW_INSTRUCTIONS


def get_tool_definitions(env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Return MCP tool definitions filtered by `CC_AGY_TOOL_PROFILE` (`core` default in strict mode, else `full`)."""
    active_env = os.environ if env is None else env
    raw_profile = (active_env.get("CC_AGY_TOOL_PROFILE") or "").strip().lower()
    if not raw_profile:
        profile = "core" if is_strict_mode_enabled(active_env) else "full"
    else:
        profile = raw_profile
    if profile in ("core", "compact"):
        core_set = set(CORE_TOOL_NAMES)
        return [_compact_core_tool_def(t) for t in TOOL_DEFINITIONS if t["name"] in core_set]
    return TOOL_DEFINITIONS


def parse_bool(value: Any, default: bool = False) -> bool:
    """Safely parse boolean tool arguments, treating string 'false'/'0'/'no' as False."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ("false", "0", "no", "off", ""):
            return False
        if norm in ("true", "1", "yes", "on"):
            return True
    raise ValueError(f"Expected boolean value, got {value!r}")


class InvalidParamsError(ValueError):
    """Raised when MCP tool call parameters fail schema/type validation (-32602)."""


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidParamsError(f"'{field_name}' must be a non-empty string")
    return value


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidParamsError(f"'{field_name}' must be a string, got {type(value).__name__}")
    return value


def _require_str_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise InvalidParamsError(f"'{field_name}' must be a list of non-empty strings")
    return list(value)


def _parse_positive_float(value: Any, field_name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise InvalidParamsError(f"'{field_name}' must be a positive number, got bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidParamsError(f"'{field_name}' must be a valid number: {exc}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise InvalidParamsError(f"'{field_name}' must be a finite number > 0, got {value!r}")
    return parsed


def _parse_optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidParamsError(f"'{field_name}' must be a finite number, got bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidParamsError(f"'{field_name}' must be a valid number: {exc}") from exc
    if not math.isfinite(parsed):
        raise InvalidParamsError(f"'{field_name}' must be a finite number, got {value!r}")
    return parsed


def _parse_positive_int(value: Any, field_name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise InvalidParamsError(f"'{field_name}' must be a positive integer, got bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidParamsError(f"'{field_name}' must be a valid integer: {exc}") from exc
    if parsed <= 0:
        raise InvalidParamsError(f"'{field_name}' must be an integer > 0, got {value!r}")
    return parsed


def _parse_optional_nonneg_int(value: Any, field_name: str) -> int | None:
    """None keeps the plugin default; 0 disables the cap; otherwise a positive integer."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidParamsError(f"'{field_name}' must be an integer, got bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidParamsError(f"'{field_name}' must be a valid integer: {exc}") from exc
    if parsed < 0:
        raise InvalidParamsError(f"'{field_name}' must be >= 0, got {value!r}")
    return parsed


def _validate_optional_str_list(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise InvalidParamsError(f"'{field_name}' must be a list of strings")
    return value


class MCPServer:
    """JSON-RPC 2.0 MCP server dispatching Vertex AI tools."""

    def __init__(
        self,
        job_manager: JobManager | None = None,
        store: ReceiptStore | None = None,
        ask_backend: Any = None,
        batch_writer_factory: Any = None,
    ) -> None:
        self.store = store or (job_manager.store if job_manager else ReceiptStore())
        self.job_manager = job_manager or JobManager(store=self.store)
        self.ask_backend = ask_backend
        if ask_backend is not None and getattr(self.job_manager, "ask_backend", None) is None:
            self.job_manager.ask_backend = ask_backend
        self.batch_writer_factory = batch_writer_factory
        # Set by serve_stdio: writes a JSON-RPC notification (e.g. notifications/progress) to the client.
        self.notify: Any = None
        # notifications/cancelled flips the stop event of the matching in-flight batch request.
        self._active_stops: dict[Any, threading.Event] = {}
        self._cancel_lock = threading.Lock()

    def _progress_reporter(self, progress_token: Any, heartbeat_seconds: float = 30.0):
        """Return (report, stop) for MCP progress: report(done, total, msg) plus a heartbeat thread."""
        if progress_token is None or self.notify is None:
            return None, (lambda: None)
        state = {"done": 0, "total": None, "message": "starting"}
        lock = threading.Lock()
        halt = threading.Event()

        def send() -> None:
            params: dict[str, Any] = {"progressToken": progress_token, "progress": state["done"], "message": state["message"]}
            if state["total"] is not None:
                params["total"] = state["total"]
            try:
                self.notify({"jsonrpc": "2.0", "method": "notifications/progress", "params": params})
            except Exception:  # noqa: BLE001 - progress is best effort
                pass

        def report(done: int, total: int, message: str) -> None:
            with lock:
                state.update(done=done, total=total, message=message)
                send()

        def beat() -> None:
            while not halt.wait(heartbeat_seconds):
                with lock:
                    state["message"] = f"working ({state['done']} settled)"
                    send()

        threading.Thread(target=beat, name="cc-agy-progress", daemon=True).start()
        return report, halt.set

    def handle_Message(self, message: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Process a JSON-RPC 2.0 message (or batch) and return a response dict (or None for notifications)."""
        if isinstance(message, list):
            if not message:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: empty batch"},
                }
            responses = []
            for item in message:
                resp = self._handle_single_message(item)
                if resp is not None:
                    responses.append(resp)
            return responses if responses else None

        return self._handle_single_message(message)

    handle_message = handle_Message

    def _handle_single_message(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": f"Invalid Request: expected JSON object, got {type(message).__name__}",
                },
            }

        method = message.get("method")
        msg_id = message.get("id")
        has_id = "id" in message and msg_id is not None

        if not isinstance(method, str) or not method.strip():
            return {
                "jsonrpc": "2.0",
                "id": msg_id if has_id else None,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: missing or non-string 'method'",
                },
            }

        params = message.get("params")
        if params is not None and not isinstance(params, dict):
            if not has_id:
                return None
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params: 'params' must be an object",
                },
            }
        params = params or {}

        # Notifications have no id and expect no response
        if not has_id:
            if method == "notifications/cancelled":
                with self._cancel_lock:
                    stop = self._active_stops.get(params.get("requestId"))
                if stop is not None:
                    stop.set()
            return None

        try:
            if method == "initialize":
                requested_version = str(params.get("protocolVersion") or "")
                negotiated = (
                    requested_version
                    if requested_version in SUPPORTED_MCP_VERSIONS
                    else MCP_PROTOCOL_VERSION
                )
                init_result: dict[str, Any] = {
                    "protocolVersion": negotiated,
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "cc-agy-plugin-vertex",
                        "version": "0.3.0",
                    },
                }
                host_instructions = server_instructions_for_host()
                if host_instructions:
                    init_result["instructions"] = host_instructions
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": init_result,
                }

            if method == "ping":
                return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": get_tool_definitions()},
                }

            if method == "tools/call":
                tool_name = str(params.get("name") or "")
                if tool_name not in KNOWN_TOOL_NAMES:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32602,
                            "message": f"Invalid params: unknown tool {tool_name!r}",
                        },
                    }
                raw_args = params.get("arguments")
                if raw_args is not None and not isinstance(raw_args, dict):
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params: 'arguments' must be an object",
                        },
                    }
                arguments = dict(raw_args or {})
                meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
                payload = self._dispatch_tool(tool_name, arguments, progress_token=meta.get("progressToken"),
                                              request_id=msg_id)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(payload, indent=2, sort_keys=True),
                            }
                        ],
                        "isError": bool(payload.get("is_error", False)),
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }
        except InvalidParamsError as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32602,
                    "message": f"Invalid params: {exc}",
                },
            }
        except BudgetExceededError as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(exc.as_dict(), indent=2),
                        }
                    ],
                    "isError": True,
                },
            }
        except (Exception, asyncio.CancelledError) as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                indent=2,
                            ),
                        }
                    ],
                    "isError": True,
                },
            }

    def _batch_auth(self, args: dict[str, Any], model: str) -> dict[str, Any]:
        return resolve_auth_config(
            requested_project=_optional_str(args.get("project"), "project"),
            requested_location=_optional_str(args.get("location"), "location"),
            requested_api_key=_optional_str(args.get("api_key"), "api_key"),
            model=model,
        )

    def _run_file_batch(self, plan: Any, args: dict[str, Any], progress_token: Any, *, request_id: Any = None,
                        **runner_kwargs: Any) -> dict[str, Any]:
        try:
            settings = resolve_batch_settings(args)
        except ValueError as exc:
            raise InvalidParamsError(str(exc)) from exc
        auth = self._batch_auth(args, settings.model)
        report, stop_beat = self._progress_reporter(progress_token)
        stop = threading.Event()
        if request_id is not None:
            with self._cancel_lock:
                self._active_stops[request_id] = stop
        try:
            runner = FileBatchRunner(self.job_manager, writer_factory=self.batch_writer_factory, progress=report, stop=stop)
            return runner.run(plan, settings, auth, **runner_kwargs)
        finally:
            stop_beat()
            if request_id is not None:
                with self._cancel_lock:
                    self._active_stops.pop(request_id, None)

    def _dispatch_tool(self, name: str, args: dict[str, Any], *, progress_token: Any = None,
                       request_id: Any = None) -> dict[str, Any]:
        if name == "delegate_file_batch":
            if parse_bool(args.get("in_place"), False):
                raise InvalidParamsError(
                    "delegate_file_batch always writes into an isolated job copy; land it with delegate_apply."
                )
            allowed = validate_allowed_paths(args.get("allowed_paths")) if args.get("allowed_paths") else None
            ws = self.job_manager.workspace_root
            extra = _validate_optional_str_list(args.get("acceptance"), "acceptance") or []
            try:
                if args.get("plan_path"):
                    plan = load_batch_plan(_require_str(args.get("plan_path"), "plan_path"), ws,
                                           allowed_paths=allowed, extra_acceptance=extra)
                elif isinstance(args.get("tasks"), list):
                    plan = validate_batch_plan(
                        {"tasks": args["tasks"], "shared_context": args.get("shared_context") or [],
                         "acceptance": extra, "formatter": args.get("formatter") or ""},
                        ws, allowed_paths=allowed,
                    )
                else:
                    raise InvalidParamsError("Pass plan_path (preferred) or tasks.")
            except BatchPlanError as exc:
                return {"status": "invalid_plan", **exc.as_dict(), "is_error": True}
            return self._run_file_batch(
                plan, args, progress_token, allowed_paths=allowed, request_id=request_id,
                resume_job_id=_optional_str(args.get("resume_job_id"), "resume_job_id"),
            )

        if name == "batch_task_log":
            return batch_task_log(self.store, _require_str(args.get("job_id"), "job_id"),
                                  _require_str(args.get("task_id"), "task_id"))

        if name in ("draft_document", "write_tests_from_cases", "replicate_edit"):
            ws = Path(self.job_manager.workspace_root)
            try:
                if name == "draft_document":
                    plan, note, gate = prepare_document(
                        ws, path=_require_str(args.get("path"), "path"), purpose=str(args.get("purpose") or ""),
                        audience=str(args.get("audience") or ""), sections=args.get("sections") or [],
                        facts=args.get("facts") or [], context_files=args.get("context_files") or [],
                    )
                    edits_only = False
                elif name == "write_tests_from_cases":
                    plan, note, gate = prepare_tests(
                        ws, test_path=_require_str(args.get("test_path"), "test_path"),
                        target_path=_require_str(args.get("target_path"), "target_path"),
                        cases=args.get("cases") or [], test_command=str(args.get("test_command") or ""),
                    )
                    edits_only = False
                else:
                    plan, note, gate = prepare_replication(
                        ws, example_path=_require_str(args.get("example_path"), "example_path"),
                        targets=args.get("targets") or [], note=str(args.get("note") or ""),
                        check_command=str(args.get("check_command") or ""),
                    )
                    edits_only = True
            except AssistRequestError as exc:
                return exc.as_dict()
            return self._run_file_batch(
                plan, args, progress_token, request_id=request_id, kind=name, content_gate=gate, shared_note=note,
                edits_only=edits_only,
            )

        if name == "doctor":
            return run_doctor(
                workspace_root=self.job_manager.workspace_root,
                store=self.store,
            )

        if name == "probe_writers":
            try:
                settings = resolve_batch_settings(args)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            try:
                auth = self._batch_auth(args, settings.model)
            except ValueError as exc:
                return {"status": "error", "is_error": True, "writers": [], "error": str(exc)}
            return probe_writers(settings, auth, writer_factory=self.batch_writer_factory)

        if name == "ask_gemini":
            prompt_str = _require_str(args.get("prompt"), "prompt")
            ctx_files = _validate_optional_str_list(args.get("context_files"), "context_files")
            temp_val = _parse_optional_finite_float(args.get("temperature"), "temperature")
            try:
                scrub_b = parse_bool(args.get("scrub_receipt"), False)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            return ask_gemini(
                prompt=prompt_str,
                model=_optional_str(args.get("model"), "model"),
                context_files=ctx_files,
                system_instruction=_optional_str(args.get("system_instruction"), "system_instruction"),
                temperature=temp_val,
                thinking_level=_optional_str(args.get("thinking_level"), "thinking_level"),
                api_key=_optional_str(args.get("api_key"), "api_key"),
                project=_optional_str(args.get("project"), "project"),
                location=_optional_str(args.get("location"), "location"),
                workspace_root=self.job_manager.workspace_root,
                store=self.store,
                scrub_receipt=scrub_b,
                backend=self.ask_backend,
            )

        if name == "delegate_start":
            prompt_str = _require_str(args.get("prompt"), "prompt")
            enabled_t = _validate_optional_str_list(args.get("enabled_tools"), "enabled_tools")
            disabled_t = _validate_optional_str_list(args.get("disabled_tools"), "disabled_tools")
            spec_f = _validate_optional_str_list(args.get("spec_files"), "spec_files")
            context_f = _validate_optional_str_list(args.get("context_files"), "context_files")
            checklist_items = _validate_optional_str_list(args.get("checklist"), "checklist")
            qa_review_model = _optional_str(args.get("qa_review_model"), "qa_review_model") or (
                os.environ.get("CC_AGY_QA_REVIEW_MODEL", "").strip() or None
            )
            try:
                allowed = validate_allowed_paths(args.get("allowed_paths"))
                plan_steps = validate_plan(args.get("plan"))
                acceptance_cmds = validate_acceptance(args.get("acceptance"), self.job_manager.workspace_root)
                enable_sub = parse_bool(args.get("enable_subagents"), False)
                scrub_b = parse_bool(args.get("scrub_receipt"), False)
                wait_b = parse_bool(args.get("wait"), False)
                in_place_b = parse_bool(args.get("in_place"), False)
                preload_b = parse_bool(args.get("preload_files"), True)
                auto_apply_arg = (
                    parse_bool(args.get("auto_apply"), False)
                    if "auto_apply" in args and args.get("auto_apply") is not None
                    else None
                )
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            prompt_str, plan_steps, repair_meta = self._maybe_repair_dictation(
                prompt_str, plan_steps, checklist_items
            )
            if "checklist" in repair_meta:
                checklist_items = repair_meta["checklist"]
            if in_place_b and auto_apply_arg is False:
                raise InvalidParamsError(
                    "in_place=True cannot be combined with auto_apply=False because in_place edits "
                    "are applied directly to the working tree."
                )
            timeout_sec = _parse_positive_float(args.get("timeout_seconds"), "timeout_seconds", 1200.0)
            req_api_key = _optional_str(args.get("api_key"), "api_key")
            req_project = _optional_str(args.get("project"), "project")
            req_location = _optional_str(args.get("location"), "location")
            will_run_qa = bool(wait_b and qa_review_model and qa_review_model.strip())
            digest = self.job_manager.delegate_start(
                prompt=prompt_str,
                model=_optional_str(args.get("model"), "model"),
                allowed_paths=allowed,
                enabled_tools=enabled_t,
                disabled_tools=disabled_t,
                enable_subagents=enable_sub,
                scrub_receipt=scrub_b,
                timeout_seconds=timeout_sec,
                wait=wait_b,
                in_place=in_place_b,
                api_key=req_api_key,
                project=req_project,
                location=req_location,
                thinking_level=_optional_str(args.get("thinking_level"), "thinking_level"),
                plan=plan_steps,
                acceptance=acceptance_cmds,
                max_tool_calls=_parse_optional_nonneg_int(args.get("max_tool_calls"), "max_tool_calls"),
                max_total_tokens=_parse_optional_nonneg_int(args.get("max_total_tokens"), "max_total_tokens"),
                preload_files=preload_b,
                auto_apply=False if will_run_qa else auto_apply_arg,
                spec_files=spec_f,
                context_files=context_f,
                checklist=checklist_items,
                _skip_balanced_qa=will_run_qa,
            )
            if will_run_qa and qa_review_model:
                digest = self._maybe_run_qa_review(
                    digest,
                    prompt_str=prompt_str,
                    qa_review_model=qa_review_model,
                    in_place_b=in_place_b,
                    timeout_sec=timeout_sec,
                    req_api_key=req_api_key,
                    req_project=req_project,
                    req_location=req_location,
                    scrub_b=scrub_b,
                )
                digest = self.job_manager._maybe_auto_apply_digest(
                    str(digest["job_id"]),
                    digest,
                    auto_apply=auto_apply_arg,
                    in_place=in_place_b,
                )
            if repair_meta.get("repaired") and isinstance(digest, dict):
                digest["dictation_repaired"] = True
                if repair_meta.get("snippet_paths"):
                    digest["dictated_snippet_paths"] = repair_meta["snippet_paths"]
            return digest

        if name == "delegate_status":
            limit_val = _parse_positive_int(args.get("limit"), "limit", 20)
            return self.job_manager.delegate_status(
                job_id=_optional_str(args.get("job_id"), "job_id"),
                limit=limit_val,
            )

        if name == "delegate_result":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            try:
                wait_b = parse_bool(args.get("wait"), True)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            timeout_val = _parse_positive_float(args.get("timeout"), "timeout", 240.0)
            return self.job_manager.delegate_result(
                job_id=job_id_str,
                wait=wait_b,
                timeout=timeout_val,
            )

        if name == "get_diff":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            page_val = _parse_positive_int(args.get("page"), "page", 1)
            page_tok_val = _parse_positive_int(args.get("page_tokens"), "page_tokens", 4000)
            focus_syms = _validate_optional_str_list(args.get("focus_symbols"), "focus_symbols")
            try:
                summary_only_b = parse_bool(args.get("summary_only"), False)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            return self.job_manager.get_diff(
                job_id=job_id_str,
                file_filter=_optional_str(args.get("file_filter"), "file_filter"),
                page=page_val,
                page_tokens=page_tok_val,
                summary_only=summary_only_b,
                focus_symbols=focus_syms,
            )

        if name == "delegate_apply":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            try:
                cleanup_b = parse_bool(args.get("cleanup_worktree"), True)
                allow_scope_b = parse_bool(args.get("allow_scope_violations"), False)
                inc_unverified_tests_b = parse_bool(args.get("include_unverified_new_tests"), False)
                inc_paths = (
                    _require_str_list(args.get("include_paths"), "include_paths")
                    if "include_paths" in args and args.get("include_paths") is not None
                    else None
                )
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            return self.job_manager.delegate_apply(
                job_id=job_id_str,
                cleanup_worktree=cleanup_b,
                allow_scope_violations=allow_scope_b,
                include_paths=inc_paths,
                include_unverified_new_tests=inc_unverified_tests_b,
            )

        if name == "delegate_continue":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            followup_str = _require_str(args.get("follow_up_prompt"), "follow_up_prompt")
            try:
                wait_b = parse_bool(args.get("wait"), True)
                force_b = parse_bool(args.get("force_retry"), False)
                cont_auto_apply = (
                    parse_bool(args.get("auto_apply"), False)
                    if "auto_apply" in args and args.get("auto_apply") is not None
                    else None
                )
                cont_allowed = (
                    validate_allowed_paths(args.get("allowed_paths"))
                    if "allowed_paths" in args and args.get("allowed_paths") is not None
                    else None
                )
                cont_acceptance = (
                    validate_acceptance(args.get("acceptance"), self.job_manager.workspace_root)
                    if "acceptance" in args and args.get("acceptance") is not None
                    else None
                )
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            followup_str, _, repair_meta = self._maybe_repair_dictation(followup_str, None, None)
            timeout_sec = _parse_positive_float(args.get("timeout_seconds"), "timeout_seconds", 600.0)
            digest = self.job_manager.delegate_continue(
                job_id=job_id_str,
                follow_up_prompt=followup_str,
                allowed_paths=cont_allowed,
                acceptance=cont_acceptance,
                timeout_seconds=timeout_sec,
                wait=wait_b,
                api_key=_optional_str(args.get("api_key"), "api_key"),
                project=_optional_str(args.get("project"), "project"),
                location=_optional_str(args.get("location"), "location"),
                force_retry=force_b,
                auto_apply=cont_auto_apply,
            )
            if repair_meta.get("repaired") and isinstance(digest, dict):
                digest["dictation_repaired"] = True
                if repair_meta.get("snippet_paths"):
                    digest["dictated_snippet_paths"] = repair_meta["snippet_paths"]
            return digest

        if name == "delegate_revert":
            job_id_str = _require_str(args.get("job_id") or args.get("receipt_id"), "job_id")
            try:
                dry_b = parse_bool(args.get("dry_run"), False)
                skip_b = parse_bool(args.get("skip_dirty"), False)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            return self.job_manager.delegate_revert(job_id=job_id_str, dry_run=dry_b, skip_dirty=skip_b)

        if name == "cost_report":
            def _opt_epoch(key: str) -> float | None:
                val = args.get(key)
                if val is None:
                    return None
                try:
                    return float(val)
                except (TypeError, ValueError) as exc:
                    raise InvalidParamsError(f"{key} must be a number") from exc

            return true_total_report(
                self.store,
                self.job_manager.workspace_root,
                since_epoch=_opt_epoch("since_epoch"),
                until_epoch=_opt_epoch("until_epoch"),
                transcripts_dir=_optional_str(args.get("transcripts_dir"), "transcripts_dir"),
            )

        if name == "delegate_cancel":
            job_id_str = _require_str(args.get("job_id"), "job_id")
            try:
                cleanup_b = parse_bool(args.get("cleanup_worktree"), True)
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            return self.job_manager.delegate_cancel(
                job_id=job_id_str,
                cleanup_worktree=cleanup_b,
            )

        if name == "delegate_to_gemini":
            prompt_str = _require_str(args.get("prompt"), "prompt")
            enabled_t = _validate_optional_str_list(args.get("enabled_tools"), "enabled_tools")
            disabled_t = _validate_optional_str_list(args.get("disabled_tools"), "disabled_tools")
            qa_review_model = _optional_str(args.get("qa_review_model"), "qa_review_model") or (
                os.environ.get("CC_AGY_QA_REVIEW_MODEL", "").strip() or None
            )
            try:
                allowed = validate_allowed_paths(args.get("allowed_paths"))
                plan_steps = validate_plan(args.get("plan"))
                acceptance_cmds = validate_acceptance(args.get("acceptance"), self.job_manager.workspace_root)
                enable_sub = parse_bool(args.get("enable_subagents"), False)
                scrub_b = parse_bool(args.get("scrub_receipt"), False)
                auto_apply = parse_bool(args.get("auto_apply"), True)
                in_place_b = parse_bool(args.get("in_place"), False)
                preload_b = parse_bool(args.get("preload_files"), True)
                dtg_checklist = _validate_optional_str_list(args.get("checklist"), "checklist")
            except ValueError as exc:
                raise InvalidParamsError(str(exc)) from exc
            prompt_str, plan_steps, repair_meta = self._maybe_repair_dictation(
                prompt_str, plan_steps, dtg_checklist
            )
            if "checklist" in repair_meta:
                dtg_checklist = repair_meta["checklist"]
            if in_place_b and not auto_apply:
                raise InvalidParamsError(
                    "in_place=True cannot be combined with auto_apply=False because in_place edits "
                    "are applied directly to the working tree."
                )
            timeout_sec = _parse_positive_float(args.get("timeout_seconds"), "timeout_seconds", 1200.0)
            req_api_key = _optional_str(args.get("api_key"), "api_key")
            req_project = _optional_str(args.get("project"), "project")
            req_location = _optional_str(args.get("location"), "location")
            digest = self.job_manager.delegate_start(
                prompt=prompt_str,
                model=_optional_str(args.get("model"), "model"),
                allowed_paths=allowed,
                enabled_tools=enabled_t,
                disabled_tools=disabled_t,
                enable_subagents=enable_sub,
                scrub_receipt=scrub_b,
                timeout_seconds=timeout_sec,
                wait=True,
                in_place=in_place_b,
                api_key=req_api_key,
                project=req_project,
                location=req_location,
                thinking_level=_optional_str(args.get("thinking_level"), "thinking_level"),
                plan=plan_steps,
                acceptance=acceptance_cmds,
                max_tool_calls=_parse_optional_nonneg_int(args.get("max_tool_calls"), "max_tool_calls"),
                max_total_tokens=_parse_optional_nonneg_int(args.get("max_total_tokens"), "max_total_tokens"),
                preload_files=preload_b,
                auto_apply=False,
                checklist=dtg_checklist,
                _skip_balanced_qa=bool(qa_review_model and qa_review_model.strip()),
            )
            if qa_review_model and qa_review_model.strip():
                digest = self._maybe_run_qa_review(
                    digest,
                    prompt_str=prompt_str,
                    qa_review_model=qa_review_model,
                    in_place_b=in_place_b,
                    timeout_sec=timeout_sec,
                    req_api_key=req_api_key,
                    req_project=req_project,
                    req_location=req_location,
                    scrub_b=scrub_b,
                )
            digest = self.job_manager._maybe_auto_apply_digest(
                str(digest["job_id"]),
                digest,
                auto_apply=auto_apply,
                in_place=in_place_b,
            )
            if digest.get("status") not in ("ok", "checkpoint", "needs_input", "spec_conflict"):
                digest["is_error"] = True
            if repair_meta.get("repaired") and isinstance(digest, dict):
                digest["dictation_repaired"] = True
                if repair_meta.get("snippet_paths"):
                    digest["dictated_snippet_paths"] = repair_meta["snippet_paths"]
            return digest

        raise InvalidParamsError(f"Unknown tool: {name!r}")

    def _maybe_repair_dictation(
        self,
        prompt_str: str,
        plan_steps: list[str] | None,
        checklist_items: list[str] | None = None,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Auto-repair dictated code blocks across prompt, plan, and checklist when CC_AGY_REPAIR_CODE_DICTATION is enabled."""
        resolved_plan = list(plan_steps or [])
        resolved_checklist = list(checklist_items) if checklist_items is not None else None
        auto_repair = os.environ.get("CC_AGY_REPAIR_CODE_DICTATION", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not auto_repair:
            return (
                prompt_str,
                resolved_plan,
                {"repaired": False, "snippet_paths": [], "checklist": resolved_checklist},
            )
        from scripts.pre_tool_guard import lint_delegation_prompt, repair_dictated_prompt

        snippet_dir = self.store.data_dir / "dictated_snippets"
        rep_prompt = repair_dictated_prompt(prompt_str, snippet_dir=snippet_dir)
        repaired_any = bool(rep_prompt.get("repaired"))
        snippet_paths = list(rep_prompt.get("snippet_paths") or [])
        new_prompt = str(rep_prompt.get("prompt") or prompt_str)

        def _repair_items(items: list[str]) -> list[str]:
            nonlocal repaired_any
            repaired_list: list[str] = []
            for item in items:
                rep_item = repair_dictated_prompt(item, snippet_dir=snippet_dir)
                if rep_item.get("repaired"):
                    repaired_any = True
                    for sp in rep_item.get("snippet_paths") or []:
                        if sp not in snippet_paths:
                            snippet_paths.append(sp)
                repaired_list.append(str(rep_item.get("prompt") or item))
            return repaired_list

        new_plan = _repair_items(resolved_plan)
        new_checklist = _repair_items(resolved_checklist) if resolved_checklist is not None else None

        combined_parts = [new_prompt, *new_plan, *(new_checklist or [])]
        if len(combined_parts) > 1 and not lint_delegation_prompt("\n".join(combined_parts))["ok"]:
            new_prompt = new_prompt.rstrip() + "\n[... behavioral specification boundary ...]"
            new_plan = [
                s.rstrip() + "\n[... behavioral specification boundary ...]"
                for s in new_plan
            ]
            if new_checklist is not None:
                new_checklist = [
                    c.rstrip() + "\n[... behavioral specification boundary ...]"
                    for c in new_checklist
                ]
            repaired_any = True
        return (
            new_prompt,
            new_plan,
            {"repaired": repaired_any, "snippet_paths": snippet_paths, "checklist": new_checklist},
        )

    def _maybe_run_qa_review(
        self,
        digest: dict[str, Any],
        *,
        prompt_str: str,
        qa_review_model: str,
        in_place_b: bool,
        timeout_sec: float,
        req_api_key: str | None,
        req_project: str | None,
        req_location: str | None,
        scrub_b: bool,
    ) -> dict[str, Any]:
        if not (
            qa_review_model
            and qa_review_model.strip()
            and digest.get("status") in ("ok", "checkpoint")
            and digest.get("files_changed")
        ):
            return digest
        was_checkpoint = digest.get("status") == "checkpoint"
        job_id_str = str(digest["job_id"])
        diff_path = self.store.diff_path(job_id_str)
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""
        if not diff_text.strip():
            return digest
        wt_candidate = (
            self.job_manager.workspace_root
            if in_place_b
            else self.store.worktree_path(job_id_str)
        )
        target_wt = wt_candidate if wt_candidate.exists() else self.job_manager.workspace_root
        acc_results = digest.get("acceptance_results") or []
        test_out_lines = [
            f"$ {r.get('command', '')} (exit {r.get('exit_code', 0)})\n{r.get('output_tail', '')}".strip()
            for r in acc_results
            if isinstance(r, dict)
        ]
        review_bundle = build_review_evidence_dir(
            target_wt,
            task_prompt=prompt_str,
            diff_text=diff_text,
            test_output="\n\n".join(test_out_lines),
            files_changed=digest.get("files_changed") or [],
        )
        qa_diff, qa_truncated, _qa_orig = truncate_to_token_budget(diff_text, QA_REVIEW_MAX_TOKENS)
        slices_text = str(review_bundle.get("enclosing_slices") or "")
        slices_tok = int(review_bundle.get("enclosing_slices_tokens") or 0)
        slices_block = (
            f"\n\n## Enclosing Modified Functions & Classes\n{slices_text}"
            if (slices_text and not qa_truncated and (_qa_orig + slices_tok) <= QA_REVIEW_MAX_TOKENS)
            else ""
        )
        qa_prompt = (
            f"## Task Specification\n{prompt_str}\n\n"
            f"## Candidate Git Diff{' (truncated)' if qa_truncated else ''}\n```diff\n{qa_diff}\n```"
            f"{slices_block}"
        )
        qa_system = build_qa_diff_audit_system(
            verification=str(digest.get("verification") or ""),
            prompt=prompt_str,
        )
        qa_error: str | None = None
        try:
            qa_res = ask_gemini(
                prompt=qa_prompt,
                model=qa_review_model.strip(),
                system_instruction=qa_system,
                thinking_level="HIGH",
                api_key=req_api_key,
                project=req_project,
                location=req_location,
                workspace_root=self.job_manager.workspace_root,
                store=self.store,
                scrub_receipt=scrub_b,
                backend=self.ask_backend,
            )
            parsed_qa = parse_qa_audit_verdict(str(qa_res.get("text") or ""))
        except Exception as qa_exc:
            qa_error = f"{type(qa_exc).__name__}: {qa_exc}"
            qa_res = {}
            parsed_qa = {"verdict": "ERROR", "passed": None, "issues": []}
        resumed = False
        if (parsed_qa["passed"] is False) or was_checkpoint:
            if parsed_qa["issues"]:
                issues_block = "\n".join(parsed_qa["issues"])
                followup = (
                    "A QA diff audit identified the following issues against the task specification:\n"
                    f"{issues_block}\n\n"
                    f"{review_bundle['pointer_prompt']}\n\n"
                    "Fix each issue above cleanly in production code without weakening test inputs or altering "
                    "pre-existing test expectations (unless explicitly required by the specification), and verify tests pass."
                )
            elif parsed_qa["passed"] is False:
                followup = (
                    "A QA diff audit returned VERDICT: NEEDS_FIXES against the task specification.\n\n"
                    f"{review_bundle['pointer_prompt']}\n\n"
                    "Verify every requirement and edge case in the task specification, fix root causes in production "
                    "code without weakening pre-existing test expectations (unless explicitly required by the "
                    "specification), and ensure all tests pass."
                )
            else:
                followup = (
                    "You reached the tool-call checkpoint before completing the turn. "
                    f"{review_bundle['pointer_prompt']}\n\n"
                    "Finish any remaining implementation steps and verify tests pass."
                )
            cont_digest = self.job_manager.delegate_continue(
                job_id=job_id_str,
                follow_up_prompt=followup,
                timeout_seconds=min(timeout_sec, 600.0),
                wait=True,
                api_key=req_api_key,
                project=req_project,
                location=req_location,
                auto_apply=False,
                _prompt_source="plugin_qa_review",
            )
            resumed = cont_digest.get("continued", True) is not False
            digest = cont_digest
        qa_meta: dict[str, Any] = {
            "model": qa_review_model.strip(),
            "job_id": qa_res.get("job_id"),
            "verdict": parsed_qa["verdict"],
            "passed": parsed_qa["passed"],
            "issues": parsed_qa["issues"],
            "resumed": resumed,
            "review_dir": review_bundle["review_dir"],
            "pointer_prompt_tokens": review_bundle["pointer_prompt_tokens"],
            "usage": qa_res.get("usage"),
            "cost": qa_res.get("cost"),
        }
        if qa_error:
            qa_meta["error"] = qa_error
        digest["qa_review"] = qa_meta
        return digest


def serve_stdio(
    server: MCPServer | None = None,
    in_stream: BinaryIO | None = None,
    out_stream: BinaryIO | None = None,
) -> None:
    """Run the stdio JSON-RPC loop supporting concurrent tool execution and both framing modes."""
    active_server = server or MCPServer()
    reader = in_stream or sys.stdin.buffer
    writer = out_stream or sys.stdout.buffer
    write_lock = threading.Lock()
    in_flight_futures: list[Future[None]] = []
    last_framing = {"content_length": False}

    def _notify(message: dict[str, Any]) -> None:
        with write_lock:
            _write_response(writer, message, last_framing["content_length"])

    active_server.notify = _notify

    def _process_and_write(msg_obj: Any, framed: bool) -> None:
        last_framing["content_length"] = framed
        try:
            resp = active_server.handle_Message(msg_obj)
        except (Exception, asyncio.CancelledError) as exc:
            msg_id = msg_obj.get("id") if isinstance(msg_obj, dict) else None
            resp = (
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                                    indent=2,
                                ),
                            }
                        ],
                        "isError": True,
                    },
                }
                if msg_id is not None
                else None
            )
        if resp is not None:
            with write_lock:
                _write_response(writer, resp, framed)

    is_live_process_stdin = in_stream is None
    rpc_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cc-agy-rpc")
    try:
        while True:
            first_line = reader.readline()
            if not first_line:
                break

            stripped = first_line.strip()
            if not stripped:
                continue

            use_content_length = False
            if stripped.lower().startswith(b"content-length:"):
                use_content_length = True
                # Always consume remaining headers until blank line
                while True:
                    hdr = reader.readline()
                    if not hdr or hdr.strip() == b"":
                        break
                try:
                    length = int(stripped.split(b":", 1)[1].strip())
                    if length < 0:
                        raise ValueError("Negative Content-Length")
                except ValueError as exc:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Invalid Content-Length header: {exc}"},
                    }
                    with write_lock:
                        _write_response(writer, err_resp, use_content_length)
                    continue
                body = reader.read(length) if length > 0 else b""
                if length > 0 and not body:
                    break
                try:
                    raw_msg = body.decode("utf-8")
                except UnicodeDecodeError as exc:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Invalid UTF-8 payload: {exc}"},
                    }
                    with write_lock:
                        _write_response(writer, err_resp, use_content_length)
                    continue
            else:
                try:
                    raw_msg = stripped.decode("utf-8")
                except UnicodeDecodeError as exc:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Invalid UTF-8 payload: {exc}"},
                    }
                    with write_lock:
                        _write_response(writer, err_resp, use_content_length)
                    continue

            parsed_messages: list[Any] = []
            try:
                parsed_messages.append(json.loads(raw_msg))
            except json.JSONDecodeError as exc:
                # Attempt multi-JSON stream recovery (e.g. concatenated JSON objects on one line)
                decoder = json.JSONDecoder()
                idx = 0
                n_chars = len(raw_msg)
                recovered: list[Any] = []
                recovery_ok = True
                while idx < n_chars:
                    while idx < n_chars and raw_msg[idx].isspace():
                        idx += 1
                    if idx >= n_chars:
                        break
                    try:
                        obj, next_idx = decoder.raw_decode(raw_msg, idx)
                        recovered.append(obj)
                        idx = next_idx
                    except json.JSONDecodeError:
                        recovery_ok = False
                        break
                if recovery_ok and recovered:
                    parsed_messages.extend(recovered)
                else:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {exc}"},
                    }
                    with write_lock:
                        _write_response(writer, err_resp, use_content_length)
                    continue

            # Dispatch tools/call concurrently so ping, delegate_status, and delegate_cancel
            # never block behind an in-flight synchronous worker call.
            for msg in parsed_messages:
                if isinstance(msg, dict) and msg.get("method") == "tools/call":
                    fut = rpc_pool.submit(_process_and_write, msg, use_content_length)
                    # Keep only unfinished calls: a long session would otherwise hold every finished
                    # future (and its result) until shutdown.
                    in_flight_futures[:] = [f for f in in_flight_futures if not f.done()]
                    in_flight_futures.append(fut)
                else:
                    _process_and_write(msg, use_content_length)

        if is_live_process_stdin:
            # Host CLI closed stdin (e.g. session exit or cohort timeout): immediately interrupt
            # and cancel active workers instead of blocking up to 240s in rpc_pool.shutdown(wait=True).
            try:
                active_server.job_manager.mark_running_interrupted()
            except Exception:
                pass
            try:
                active_server.job_manager.shutdown(wait=False)
            except Exception:
                pass
            rpc_pool.shutdown(wait=False, cancel_futures=True)
        else:
            for fut in in_flight_futures:
                try:
                    fut.result(timeout=60.0)
                except Exception:
                    pass
            rpc_pool.shutdown(wait=True)
    finally:
        if is_live_process_stdin:
            rpc_pool.shutdown(wait=False, cancel_futures=True)


def _write_response(
    writer: BinaryIO,
    response: dict[str, Any] | list[dict[str, Any]],
    use_content_length: bool,
) -> None:
    encoded = json.dumps(response, separators=(",", ":")).encode("utf-8")
    if use_content_length:
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
        writer.write(header + encoded)
    else:
        writer.write(encoded + b"\n")
    writer.flush()
