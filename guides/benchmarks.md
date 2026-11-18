# Benchmark results

The full 26-task results behind the summary in the README.

We ran a paired cross-language benchmark on 26 real repository tasks:

- 10 Python tasks from DeepSWE repos
- 10 Go tasks from Aider Polyglot and Multi-SWE-bench `cli/cli`
- 6 TypeScript tasks from Multi-SWE-bench and DeepSWE

That is 208 runs across 8 setups, with each setup running each task once (`n=1`).

Prices per 1M tokens: Opus 5.5 at `400¢` in / `20¢` cached / `2,000¢` out, Gemini 4 Argon and Carbon at `200¢` / `20¢` / `1,200¢`, and Gemini Flash at `50¢` / `5¢` / `300¢`. "Flash" below means `gemini-flash-next`; only the `balanced` row still uses `gemini-3.8-flash`.

| Setup | Tasks solved | Hidden tests passed | Avg cost / task | vs Opus solo | Avg time | Best fit |
|:--|:--:|:--:|:--:|:--:|:--:|:--|
| Claude Code · Opus 5.5 (solo) | 22 / 26 | 806 / 891 | 87.6¢ | baseline | 4m 12s | Baseline · All-Opus in Claude Code (TypeScript generics 6 / 6 at 71.3¢) |
| Claude Code + plugin · Opus + Flash (`economy`) | 15 / 26 | 797 / 891 | 60.6¢ (22¢ + 39¢) | -31% | 5m 37s | Stay in CC, cut spend 31% (5 / 6 TS, 7 / 10 Go, 797 / 891 tests) |
| Antigravity · Flash -> Carbon Low | 18 / 26 | 810 / 891 | 74.9¢ ($0 Claude) | -14% | 7m 15s | Pure AGY 2-tier (18 / 26, 810 / 891 tests vs 806 Opus, 7 / 10 Py & Go, -14%) |
| Antigravity · Carbon Zero-Think (solo) | 20 / 26 | 846 / 891 | 93.9¢ ($0 Claude) | +7% | 7m 31s | Pure AGY Carbon 0-Think (94.9% tests vs 90.5% Opus; 9 / 10 Python, 7 / 10 Go) |
| Claude Code + plugin · Flash 3.8 -> Argon (`balanced`) | 21 / 26 | 855 / 891 | 81.1¢ (21¢ + 61¢) | -7% | 7m 45s | Highest CC + plugin pass rate (10 / 10 Python at -14% cost; Go -15%) |
| Claude Code + plugin · Opus + Argon (`quality`) | 20 / 26 | 865 / 891 | 80.0¢ (22¢ + 58¢) | -9% | 5m 37s | Highest holdout test pass rate (97.1% overall; 99.8% on Python) |
| Antigravity · Flash (solo) | 18 / 26 | 841 / 891 | 47.7¢ ($0 Claude) | -46% | 6m 39s | Pure AGY Flash + plugin harness (18 / 26, 841 / 891 tests, 9 / 10 Python -53%, 6 / 10 Go -52%) |
| Antigravity + plugin · Argon + Flash (`economy`) | 15 / 26 | 673 / 891 | 83.2¢ (49¢ + 34¢, $0 Claude) | -5% | 8m 34s | Interactive Argon Lead + Flash worker in AGY (4 / 6 TS, 7 / 10 Go, -45% Python avg vs Argon solo, -3 solves overall) |

## The short version

Claude Opus 5.5 alone costs 87.6¢ per task on this suite. Against that baseline:

- **Staying in Claude Code?** Add the plugin. Opus plans, Gemini Flash writes the code: **31% cheaper** (60.6¢ vs 87.6¢ per task). That `economy` setup solved 15 of 26 tasks against 22 for Opus alone. The `balanced` profile solved 21 of 26 at 81.1¢ (-7%).
- **Moving to Antigravity?** Gemini Flash alone solves 18 of 26 and passes **more hidden tests than Opus** (841 vs 806) at **about half the cost** (47.7¢, -46%).
- **Need the most solves on Gemini?** Gemini Carbon with zero thinking solves **20 of 26** and passes 846 hidden tests (Opus: 806), at roughly Opus price (93.9¢, +7%).
- **Opus still solves the most tasks** (22 of 26). Each setup ran each task once, so differences of 1 to 3 tasks are within normal run-to-run noise.

## Using `cc-agy-plugin` inside Antigravity: before vs. after

Installing `cc-agy-plugin` directly into Antigravity (`python3 -m server.install --host agy`) delivers four measured setups without paying any Claude Code lead overhead (`$0.00` Claude tokens):

| Antigravity scenario | Without `cc-agy-plugin` | With `cc-agy-plugin` | Measured gain |
|:--|:--|:--|:--|
| **1. `gemini-flash-next` across 26 tasks (`v10` raw vs. `v12` plugin harness)** | `7 / 26` solved (`26.9%`), `173 / 891` holdout tests (`19.4%`), `0 / 10` on Python, `103.5¢` per solved task | **`18 / 26` solved (`69.2%`)**, **`841 / 891` holdout tests (`94.4%`)**, **`9 / 10` on Python**, `47.7¢` avg (`68.9¢` per solve) | **+11 tasks (+157% solves), +668 tests, -33% cost per solve** (`-49%` avg cost vs. standalone Carbon Zero-Think's `93.9¢`, and **`-61%` on Python** at `72.4¢` vs. `186.1¢`) |
| **2. 2-tier `Flash -> Carbon Low` (`v12`) vs. standalone `Carbon Zero-Think` (`v9`, 26 tasks)** | `93.9¢` avg / task (`186.1¢` on Python, `45.7¢` on Go, `7 / 10` Go solves) | **`74.9¢` avg / task** (`118.7¢` on Python, **`32.6¢` on Go**, `7 / 10` Go solves, `5 / 5` Aider Go at `3.1¢`), `18 / 26` solved (`810 / 891` tests = `90.9%`) | **-20% average cost across 26 tasks** (`-36%` on Python, **`-29%` on Go**, and `-80%` on Aider Go) by keeping first-pass edits on Flash and stepping up to Carbon Low only when checks fail |
| **3. `Antigravity · Argon` Lead delegating to isolated worker vs. standalone `Argon` (16 Python DeepSWE tasks)** | `10 / 16` solved (`62.5%`), `580` total turns, `3,084¢` total (`192.8¢` / task, `308.4¢` / solve) | **`11–12 / 16` solved (`68.8%–75.0%`)**, `433` total turns, **`2,352¢` total (`147.0¢` / task**, `196.0¢–213.8¢` / solve) | **-24% total cost, -30% to -36% cost per solve, -25% turns, and +1 to +2 extra tasks solved** purely from isolated git worktree delegation and Straitjacket log containment |
| **4. `Antigravity · Argon` Lead + Flash worker (`economy` `v13`) vs. standalone `Argon` (`docs/live_r1_iter2`, `docs/live_r6_go`, `docs/live_r7_ts`, 2026-10-21..23, 26 tasks)** | `18 / 26` solved (`69.2%`), `785 / 891` holdout tests, `81.1¢` avg / task (`155.0¢` Python [`193.8¢`/solve], `27.1¢` Go, `47.8¢` TypeScript), `5m 17s` avg | **`15 / 26` solved (`57.7%`)**, `673 / 891` holdout tests, **`83.2¢` avg / task** (`49.0¢` Lead + `34.2¢` Worker, `144.2¢` / solve; **`85.3¢` on Python** [`213.3¢`/solve], `62.3¢` on Go, `114.5¢` on TypeScript [`4 / 6`]), `8m 34s` avg | **-45% avg cost on Python DeepSWE** (`85.3¢` vs. `155.0¢`, `+10%` per solve) and **+1 extra TypeScript solve** (`4 / 6` vs. `3 / 6`, adding `vuejs-10161` while retaining `darkreader-6747`, `darkreader-7241`, and `vuejs-11694`), with **-3 solves (`15 / 26` vs. `18 / 26`, -112 net holdout tests: -115 across 4 short-Lead-loop Python runs [`2-7` Lead turns, `4.1¢-10.3¢` Lead each], +15 on `tomlkit`, -12 on `koota`) where the run failed the new holdout suite (`aiomonitor` `2/54`, `psd-tools` `7/45`, `mobly` `61/83`, `httpx` `119/122`)**, while **7 long Lead loops (`5` solved, `2` unsolved; `1,009¢` of `1,273¢` Lead spend) drove +2.6% avg cost (`+130%` Go, `+140%` TS) and +62% wall time**, making autonomous `AGY · Flash (solo)` (`47.7¢`, `18 / 26`) the cheaper Antigravity default |

## How the runs were routed and priced

All `cc-agy-plugin` worker calls, Claude Code proxy runs, and `v9`/`v10`/`v12` Antigravity runs route through free internal Cloud Code PA with `0` BAIC calls; only `v13`'s standalone `agy` CLI Argon Lead authenticated via Gemini Enterprise BAIC (`businessaicode.googleapis.com`, `0` `aiplatform.googleapis.com` calls, Vertex billing status `Unverified`), while the older standalone `AGY · Argon (solo)` baseline has no CLI logs on disk (routing `Unverified`).

> [!NOTE]
> Each setup ran once per task (`n=1`) in the 26-task cross-language suite, so 1-to-3 task solve differences sit within single-run variance (`v13`'s -3 net solves vs. standalone Argon solo came from 4 short-Lead-loop Python runs [`aiomonitor` `2/54`, `psd-tools` `7/45`, `mobly` `61/83`, `httpx` `119/122`: `2-7` Lead turns] where the run failed the new holdout suite, offset by `+1` TypeScript solve [`vuejs-10161`], while multi-turn Lead loops on 7 long tasks drove the extra cost and wall time). All Flash worker calls, Claude Code proxy runs, and `v9`/`v10`/`v12` Antigravity runs are wire-reconciled against free internal Cloud Code PA token logs with `0` BAIC calls and `0` post-exit orphan calls (on `cli__cli-6578`, the `v13` Argon Lead ran `rm -rf api/queries_user_test.go .cc-agy-data` before receipt copy, so its 37 worker calls / `37.06¢` were backfilled directly from the CCPA wire log with `0.00¢` delta by construction). Only `v13`'s standalone `agy` CLI Argon Lead ran over Gemini Enterprise BAIC (`businessaicode.googleapis.com`, `0` `aiplatform.googleapis.com` calls, Vertex billing status `Unverified`) and is priced from transcripts (`374` priced turns out of `405` BAIC HTTP requests; `31` requests have no matching usage record), while the older standalone `AGY · Argon (solo)` baseline (`live_r1_iter2`, `live_r6_go`, `live_r7_ts`) has no CLI logs on disk (routing `Unverified`).

[Back to the README](../README.md)
