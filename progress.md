# OpenPI LIBERO Eval Migration Progress

## Goal
- Port the full OpenVLA-style LIBERO evaluation control logic into `openpi` while keeping existing files and flows untouched.

## Scope and Isolation Strategy
- Created a **new isolated module** under `examples/libero/openvla_eval_port/`.
- Did **not** modify existing `examples/libero/main.py` or serving code.
- New files are opt-in and only used when explicitly executed.

## Completed Changes

### 1) New isolated package
- Added `examples/libero/openvla_eval_port/__init__.py`.

### 2) LLM replanning module
- Added `examples/libero/openvla_eval_port/libero_replanner.py`.
- Implemented:
  - frame sampling + image encoding to data URLs
  - strict JSON prompt generation
  - OpenAI-compatible multimodal API call
  - robust JSON parsing + primitive validation
  - retry mechanism
  - fallback recovery plan (`init()`)
- Security hardening:
  - removed hardcoded keys and endpoints
  - now reads:
    - `OPENPI_REPLAN_API_KEY`
    - `OPENPI_REPLAN_API_BASE_URL` (optional, defaults to `http://127.0.0.1:3000/v1`)

### 3) Subtask engine module
- Added `examples/libero/openvla_eval_port/libero_subtask.py`.
- Implemented:
  - Libero-10 subtask plans
  - primitive rules (`reach/grasp/release/move/rotate/push/flip/insert/press/contact/turn_on/open/predicate/init/hold`)
  - `SubtaskTracker` with:
    - env-driven update
    - manual advance
    - rollback
    - recovery-plan injection for LLM replanning
    - status/history tracking

### 4) Full evaluation runner for OpenPI
- Added `examples/libero/openvla_eval_port/run_libero_eval_openpi.py`.
- Implemented OpenVLA-style full pipeline on top of websocket inference:
  - LIBERO suite/task/episode loops
  - initial-state loading (`DEFAULT` or JSON file)
  - subtask-driven prompting (`current_instruction`)
  - action chunk queue + open-loop execution
  - hardcoded primitive handling for `init` / `hold`
  - two switch modes:
    - `env` (rule-based completion)
    - `progress` (threshold-based completion)
  - progress rollback / stall detection
  - optional LLM replanning + recovery plan injection
  - per-task and global metrics
  - rollout video dumping
  - file + console logging

## Notes / Compatibility
- In `progress` mode, this evaluator expects server inference output to include either:
  - key from `--progress-key` (default `progress`), or
  - `progress_pred`.
- If server does not provide progress, use `--subtask-switch-mode env` (fully supported).

## How to Run

1. Start policy server (existing flow):
- `uv run scripts/serve_policy.py --env LIBERO`

2. Run new evaluator:
- `python examples/libero/openvla_eval_port/run_libero_eval_openpi.py --task-suite-name libero_10 --num-trials-per-task 10`

3. Enable LLM replanning (optional):
- set `OPENPI_REPLAN_API_KEY`
- optional `OPENPI_REPLAN_API_BASE_URL`
- run with `--use-llm-replan true`

## Pending / Optional Follow-ups
- If needed, extend server output schema to always return progress for stable `progress` mode.
- Optional: add text overlay rendering directly onto rollout frames.
- Optional: add state-pool export support if curriculum tooling is required in OpenPI.

