# Transformer Character LM — *Testbed D*

Minimal **character-level Transformer** workload for comparing **Kourkoutas-β** against Adam on Apple MLX.

> This directory is self-contained. It only requires a small text corpus (e.g. a 5–10 MB “small-enwik8” file) and MLX + `kbeta` installed in editable mode.

---

## Files

- `testbed_d.py` — training script / entry point

---

## Setup

From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

MLX automatically detects Apple Silicon; no device flags are needed.

---

## Data

Place a small UTF-8 plain-text file here:

```
examples/transformer_char_lm/data/small_enwik8.txt
```

If you already have the full `enwik8`, you can slice it quickly:

```bash
mkdir -p examples/transformer_char_lm/data
head -c 5000000 /path/to/enwik8 > examples/transformer_char_lm/data/small_enwik8.txt
```

> ⚠️ Datasets are **not** committed to the repo. Keep `data/` out of version control.

---

## Usage

From this folder:

```bash
cd examples/transformer_char_lm
python testbed_d.py --help
```

### Quick smoke-tests

- **Kourkoutas-β**

```bash
python testbed_d.py   --text ./data/small_enwik8.txt   --opt kbeta   --ctx 256   --steps 3000   --lr 3e-4
```

- **Adam (baseline)**

```bash
python testbed_d.py   --text ./data/small_enwik8.txt   --opt adam   --ctx 256   --steps 3000   --lr 3e-4
```

Increase `--steps`, `--ctx`, or model size flags for longer runs.

---

## CLI flags

Extracted from `testbed_d.py`:

- **--text** `str`
- **--steps** `int` (default: `60000`)
- **--batch** `int` (default: `64`)
- **--d_model** `int` (default: `256`)
- **--n_layer** `int` (default: `6`)
- **--n_head** `int` (default: `4`)
- **--lmin** `int` (default: `64`)
- **--lmax** `int` (default: `256`)
- **--ctx** `int` (default: `256`)
- **--lr** `float` (default: `3e-4`)
- **--warmup** `int` (default: `200`)
- **--seed** `int` (default: `0`)
- **--opt** `{"kbeta","adam"}` (default: `"kbeta"`)
- **--adam_beta2** `float` (default: `0.999`)
- **--eval_every** `int` (default: `2000`)
- **--layer_bucket** `{"global","shape","per-array"}` (default: `"global"`)
- **--compile**
- **--len_bucket** `int` (default: `32`) — round L up to reduce compile churn; `1` disables
- **--lr_schedule** `str` (default: `""`) — e.g. `"0:3e-4,20000:1.5e-4,40000:1e-4,50000:1e-5"`
- **--barrier_every** `int` (default: `1`)
- **--val_frac** `float` (default: `0.1`) — fraction of text held out for validation
- **--eval_bs** `int` (default: `128`)
- **--fixed_eval_seed** `int` (default: `1234`) — freezes the eval batch
- **--deterministic** — stable per-array bucket + deterministic length draw
- **--wd** `float` (default: `0.0`) — decoupled weight decay; `0` disables
- **--wd_bias**, **--wd_norm**, **--wd_embed**
- **--early_stop_patience** `int` (default: `0`) — patience in eval checks with no improvement; `0` disables
- **--early_stop_min_delta** `float` (default: `0.0`) — min absolute improvement in eval loss
- **--early_stop_warmup** `int` (default: `0`) — number of initial eval checks to ignore

---

## MLX usage notes

- Training follows MLX idioms:

  ```python
  loss_and_grad = nn.value_and_grad(model, loss_fn)
  loss, grads = loss_and_grad(model, batch_inputs, batch_targets)
  optimizer.update(model, grads)
  ```

- Call `mx.eval(...)` on updated parameters or losses when mixing Python control flow or for per-step timing (to flush lazy computation).

- Use **Kourkoutas-β** by importing from the package:

  ```python
  from kbeta import KourkoutasSoftmaxFlex as Kbeta
  ```

- Control bucket strategy for K-β with `--layer_bucket` (`global`, `shape`, `per-array`).

---

## Results

For quick smoke-tests on a 5–10 MB corpus, training loss should decrease within a few hundred steps.
Use `--eval_every` to monitor validation loss, and `--early_stop_*` options for patience-based early stopping.

---

## Reproducibility

Example shell script for reproducing paper results:

```zsh
#!/usr/bin/env zsh
set -euo pipefail

mkdir -p logs_enwik

for seed in {1..2}; do
  echo "==> SEED $seed (kbeta)"
  python -u kbeta_char_transformerCompile8.py --text ./small-enwik8.txt     --steps 50001 --batch 4 --d_model 512 --n_layer 6 --n_head 8     --ctx 512 --lmin 16 --lmax 512 --warmup 250 --opt kbeta --adam_beta2 0.95     --layer_bucket per-array --barrier_every 100 --eval_every 500     --lr 1e-3     --seed "$seed" --fixed_eval_seed 1234 --deterministic --compile     --wd 0.0 --lr_schedule "1:1e-3,30000:5e-4,40000:1e-4,60000:1e-5"     2>&1 | tee "logs_enwik/kbeta_seed${seed}.log"

  echo "==> SEED $seed (adam β2=0.95)"
  python -u kbeta_char_transformerCompile8.py --text ./small-enwik8.txt     --steps 50001 --batch 4 --d_model 512 --n_layer 6 --n_head 8     --ctx 512 --lmin 16 --lmax 512 --warmup 250 --opt adam --adam_beta2 0.95     --layer_bucket per-array --barrier_every 100 --eval_every 500     --lr 1e-3     --seed "$seed" --fixed_eval_seed 1234 --deterministic --compile     --wd 0.0 --lr_schedule "1:1e-3,30000:5e-4,40000:1e-4,60000:1e-5"     2>&1 | tee "logs_enwik/adam95_seed${seed}.log"

  echo "==> SEED $seed (adam β2=0.999)"
  python -u kbeta_char_transformerCompile8.py --text ./small-enwik8.txt     --steps 50001 --batch 4 --d_model 512 --n_layer 6 --n_head 8     --ctx 512 --lmin 16 --lmax 512 --warmup 250 --opt adam --adam_beta2 0.999     --layer_bucket per-array --barrier_every 100 --eval_every 500     --lr 1e-3     --seed "$seed" --fixed_eval_seed 1234 --deterministic --compile     --wd 0.0 --lr_schedule "1:1e-3,30000:5e-4,40000:1e-4,60000:1e-5"     2>&1 | tee "logs_enwik/adam999_seed${seed}.log"
done
```

---

## License

This example falls under the repository’s MIT license.
