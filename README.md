# Single-observable trajectory-regularization reproduction

Public source snapshot and rerun records for the best single-observable experiment
found in the observable-regulation learning-hyperparameter sweep.

## Configuration

- Observable: `val.layer_2.attn_out.l1`
- Trajectory coefficient: `0.007`
- Trajectory lead: `100` steps
- Regularization cutoff: step `750`
- Seed: `42`
- Training steps: `2000`
- `MATRIX_LR`: `0.045`
- `NGRAM_VE_BETAS`: `(0.5, 0.9995)`
- `NGRAM_VE_LR_SCALE`: `1.2`
- `NGRAM_VE_BETA2_WARMDOWN`: `0.99995`

## Results

| Run | `val_bpb` | Improvement vs baseline | Gap vs original best |
|---|---:|---:|---:|
| Original sweep best | **0.9305633662** | 0.0038533071 | — |
| Rerun 1 | **0.9319320959** | 0.0024845775 | +0.0013687296 |
| Rerun 2 | **0.9325372960** | 0.0018793773 | +0.0019739298 |
| Rerun mean | **0.9322346959** | 0.0021819774 | +0.0016713297 |

Baseline: `0.9344166734`.

Both fresh reruns beat the baseline, but neither reproduced the original best
within `0.001 val_bpb`. This supports a repeatable directional improvement, not
stable reproduction of the exact best metric.

## Repository layout

- `source/`: exact source snapshot, lockfile, target trajectory, and original summary.
- `reproduce.py`: isolated one-command rerun launcher.
- `original_result/`: original sweep and earlier replicate records.
- `results/`: two fresh rerun summaries and logs.
- `release/`: downloadable source and result archives.
- `SHA256SUMS`: integrity manifest.

## Run

The experiment requires the same dataset/tokenizer mount and a CUDA-capable
PyTorch environment used by the original autoresearch setup.

```bash
/home/user/ph/autoresearch/.venv/bin/python reproduce.py \
  --python /home/user/ph/autoresearch/.venv/bin/python \
  --gpu 1 \
  --output-dir rerun
```

The launcher accepts device identifiers `1` through `7` and writes `train.log`
and `summary.json` under the selected output directory.

## Integrity

From the repository root:

```bash
shasum -a 256 -c SHA256SUMS
```

The source snapshot is preserved exactly; CUDA execution is not guaranteed to
be bitwise deterministic even with a fixed seed.
