#!/usr/bin/env bash
set -euo pipefail
source /data/conda_a100/ourconda_bashrc
conda activate libero
source /data/Embobrain/code/RLinf/openpi_libero/bin/activate
cd /data/Embobrain/openpi

python - <<'PY'
import glob, os, subprocess, sys

script = "/data/Embobrain/openpi/examples/convert_jax_model_to_pytorch_lora_aware_progress.py"
roots = sorted(
    p for p in glob.glob("/data/Embobrain/openpi/checkpoints/pi05_libero/pi05-full-split_padded-*-fewshot")
    if not p.endswith("_pytorch") and not p.endswith("_pytorch_progress")
)

jobs = []
for root in roots:
    steps = sorted(
        d for d in glob.glob(root + "/*") if os.path.isdir(d) and os.path.basename(d).isdigit()
    )
    steps = sorted(steps, key=lambda x: int(os.path.basename(x)))
    for s in steps:
        step = os.path.basename(s)
        out = f"{root}_pytorch_progress/{step}"
        jobs.append((s, out))

print(f"TOTAL_JOBS={len(jobs)}")
ok = 0
fail = 0
for i, (ckpt, out) in enumerate(jobs, 1):
    os.makedirs(out, exist_ok=True)
    sf = os.path.join(out, "model.safetensors")
    ph = os.path.join(out, "progress_head.pt")
    if os.path.exists(sf) and os.path.exists(ph):
        print(f"[{i}/{len(jobs)}] SKIP already_done {ckpt}")
        ok += 1
        continue

    cmd = [
        sys.executable,
        script,
        "--config_name", "pi05_libero",
        "--checkpoint_dir", ckpt,
        "--output_path", out,
        "--precision", "bfloat16",
        "--progress_variant", "auto",
    ]
    print(f"[{i}/{len(jobs)}] RUN {ckpt} -> {out}")
    try:
        subprocess.run(cmd, check=True)
        ok += 1
        print(f"[{i}/{len(jobs)}] OK {ckpt}")
    except subprocess.CalledProcessError as e:
        fail += 1
        print(f"[{i}/{len(jobs)}] FAIL {ckpt} rc={e.returncode}")

print(f"DONE ok={ok} fail={fail} total={len(jobs)}")
if fail:
    raise SystemExit(1)
PY
