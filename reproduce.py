#!/usr/bin/env python3
"""Re-run the best single-observable experiment from its exact source snapshot."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

SOURCE_FILES = (
    "train.py",
    "lib.py",
    "observable.py",
    "prepare.py",
    "data_split.json",
    "pyproject.toml",
    "uv.lock",
    "trajectory_reg_curve_val_layer_2_attn_out_l1.csv",
)


def main() -> None:
    bundle = Path(__file__).resolve().parent
    source = bundle / "source"
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/home/user/ph/autoresearch/.venv/bin/python")
    parser.add_argument("--gpu", required=True, help="CUDA device id; this bundle permits 1 through 7")
    parser.add_argument("--output-dir", type=Path, default=bundle / "rerun")
    args = parser.parse_args()
    if args.gpu not in {str(i) for i in range(1, 8)}:
        raise SystemExit("--gpu must be one of 1,2,3,4,5,6,7")

    output = args.output_dir.resolve()
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(source / name, work / name)
    shutil.rmtree(work / "run_artifacts", ignore_errors=True)
    shutil.rmtree(work / "observable_csv", ignore_errors=True)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "SEED": "42",
            "MAX_TRAIN_STEPS": "2000",
            "ATTN_ENTROPY_REG": "0",
            "TRAJECTORY_REG_OBSERVABLE": "val.layer_2.attn_out.l1",
            "TRAJECTORY_REG_CURVE_FILE": str(work / "trajectory_reg_curve_val_layer_2_attn_out_l1.csv"),
            "TRAJECTORY_REG_MODE": "trajectory",
            "TRAJECTORY_REG_COEF": "0.007",
            "TRAJECTORY_REG_UNTIL_STEP": "750",
            "TRAJECTORY_REG_LEAD_STEPS": "100",
        }
    )

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [args.python, "-u", "train.py"],
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise SystemExit(f"training exited {completed.returncode}; inspect {log_path}")

    generated = work / "run_artifacts" / "latest_rsi_training_summary.json"
    if not generated.exists():
        raise SystemExit(f"summary missing: {generated}")
    summary = json.loads(generated.read_text(encoding="utf-8"))
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"val_bpb": summary["val_bpb"], "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
