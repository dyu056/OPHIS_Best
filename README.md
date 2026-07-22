# OPHIS_Best

Reproducible source snapshot for the best OPHIS single-observable trajectory-regularization result.

## Result

Validation bits per byte (`val_bpb`; lower is better) was measured across repeated runs.

| Model | Mean `val_bpb` | Mean ± 2 sample SD |
|---|---:|---:|
| OPHIS RSI baseline | 0.9340967 | 0.9334899–0.9347035 |
| OPHIS best result | **0.9318420** | 0.9301556–0.9335284 |

The best result reduces mean `val_bpb` by **0.0022547**, equivalent to **7.43×** the baseline sample standard deviation. The ranges above describe run-to-run dispersion; they are not confidence intervals.

The experiment regularizes `val.layer_2.attn_out.l1` toward the supplied trajectory with coefficient `0.007`, a 100-step lead, and a cutoff at step 750. Each run uses seed 42 and trains for 2,000 steps.

## Reproduce the best result

Requirements: Linux, Python 3.10+, [`uv`](https://docs.astral.sh/uv/), and a CUDA-capable NVIDIA GPU. The recorded runs used about 69 GB of VRAM.

```bash
git clone https://github.com/dyu056/OPHIS_Best.git
cd OPHIS_Best

# Build the locked environment and prepare the dataset/tokenizer once.
cd source
uv sync --frozen
uv run prepare.py
cd ..

# The launcher accepts CUDA device IDs 1–7.
python reproduce.py \
  --python "$PWD/source/.venv/bin/python" \
  --gpu 1 \
  --output-dir rerun

cat rerun/summary.json
```

The launcher copies the exact source snapshot and configuration into `rerun/work`, writes the full log to `rerun/train.log`, and writes the final metrics to `rerun/summary.json`. Repeat into separate output directories to measure run-to-run variation. CUDA training is not guaranteed to be bitwise deterministic, so compare the distribution rather than expecting an identical final decimal.

To verify the bundled artifacts before running:

```bash
shasum -a 256 -c SHA256SUMS
```
