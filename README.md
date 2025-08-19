[![CI (macOS arm64)](https://github.com/sck-at-ucy/kbeta/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sck-at-ucy/kbeta/actions/workflows/ci.yml)

<p align="center">
  <img src="assets/MLX_Kourkoutas.png" width="600"/>
</p>

# kbeta – *Kourkoutas‑β Optimiser*   🌞🦎🚀📈

> Reference implementation of **Kourkoutas‑β: A Sunspike‑Driven Adam Optimizer with Desert Flair**
> Published as [arXiv:2508.12996](http://arxiv.org/abs/2508.12996).

This repository provides the optimiser implementation together with example workloads for reproducibility.

---

## Table of Contents
1. [Key ideas](#key-ideas)
2. [Project layout](#project-layout)
3. [Quick start](#quick-start)
4. [Installation](#installation)
5. [Using Kourkoutas‑β in your own model](#minimal-example)
6. [Example workloads](#example-workloads)
7. [Dataset and creation (verifiable)](#dataset-and-creation-verifiable)
8. [Model and training protocol](#model-and-training-protocol)
9. [Optimizers and settings](#optimizers-and-settings)
10. [Companion repositories](#companion-repositories)
11. [Tests & linting](#tests--linting)
12. [Citation](#citation)
13. [License](#license)
14. [Contributing & roadmap](#contributing--roadmap)

---

## Key ideas

* **Layer‑wise dynamic β₂** driven by a bounded *sun‑spike* signal (gradient norm vs. EMA).
* **Two β₂ parameters**: *β₂_min* for agility under spikes, *β₂_max* for stability when calm.
* **Optional features**: soft‑max AMSGrad, trust‑region clipping, adaptive tiny term.
* **Drop‑in compatibility**: recovers **exact Adam** when dynamic β₂ and extras are disabled.
* 100 % **Apple MLX** compatible – no PyTorch required.

See the paper for derivations, experiments, and theoretical analysis.

---

## Conceptual overview

### High‑level intuition – the “desert lizard” view
*Kourkoutas‑β* is an Adam‑style optimiser whose second‑moment decay **β₂** is no longer a hard‑wired constant.
Instead, every update computes a **sun‑spike score**—a single, cheap scalar that compares the current gradient magnitude to its exponentially‑weighted history.  We then **map that score to β₂ on the fly**:

| Sun‑spike | Lizard metaphor | Adaptive behaviour |
|-----------|-----------------|--------------------|
| **High**  | The desert sun is scorching — the lizard is “fully warmed up” and sprints. | **Lower β₂ toward β₂,min** → second‑moment memory shortens, allowing rapid, large parameter moves. |
| **Low**   | It’s cool; the lizard feels sluggish and takes cautious steps. | **Raise β₂ toward β₂,max** → longer memory, filtering noise and producing steadier updates. |

Because the sun‑spike diagnostic **exists only in Kourkoutas‑β**, the method can be viewed as *Adam with a temperature‑controlled β₂ schedule*: warm gradients trigger exploration; cooler gradients favour exploitation and stability.

---

## Project layout

```
kbeta
├── src/kbeta/                   # pip package
│   ├── __init__.py              # exports KourkoutasBeta / KourkoutasSoftmaxFlex
│   └── optim/
│       └── kbeta_softmax.py     # implementation
│
├── examples/
│   └── transformer_char_lm/     # Testbed D: character‑level LM on small‑enwik8
│
├── tests/                       # pytest suite (smoke + ablation tests)
├── assets/                      # logo and figure
├── pyproject.toml
├── MANIFEST.in
├── README.md
└── LICENSE
```

---

## Quick start

```bash
# 1. clone your fork
git clone git@github.com:<YOUR-USERNAME>/kbeta.git
cd kbeta

# 2. create a fresh virtualenv
python -m venv .venv && source .venv/bin/activate

# 3. editable install + dev extras
pip install -e ".[dev]"

# 4. run the smoke + ablation tests
pytest -q
```
---

## Installation

### Option 1: PyPI wheels (end-users)

If you only want the optimiser in your own MLX projects, install from PyPI:

```bash
pip install kbeta
```

This gives you just the `kbeta` package with the latest MLX.

For development tools and examples:

```bash
pip install "kbeta[dev]"
```

For exact reproducibility of the paper results (MLX 0.26.3, Adam-95/999 baselines):

```bash
pip install "kbeta[repro]"
```

---

### Option 2: Cloning the repo (researchers / contributors)

If you want to run the example workloads or contribute to development, clone the repo:

```bash
git clone https://github.com/sck-at-ucy/kbeta.git
cd kbeta
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the package in editable mode and makes all example scripts available.

---

## Minimal example

```python
import time
import mlx.core as mx
import mlx.nn as nn
from kbeta import KourkoutasBeta

num_features, num_examples, num_iters, lr = 100, 1000, 1000, 0.01

# True parameters and data
w_star = mx.random.normal((num_features,))
X = mx.random.normal((num_examples, num_features))
y = X @ w_star + 1e-2 * mx.random.normal((num_examples,))

# Simple model with one parameter
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.zeros((num_features,))

    def __call__(self, x):
        return x @ self.w

model = Model()

def loss_fn(m):
    return 0.5 * mx.mean(mx.square(m(X) - y))

opt = KourkoutasBeta(learning_rate=lr)
opt.init(model.parameters())

grad_fn = nn.value_and_grad(model,loss_fn)

tic = time.time()
for _ in range(num_iters):
    loss, grads = grad_fn(model)
    opt.update(model, grads)
    mx.eval(model.parameters())
toc = time.time()

error_norm = float(mx.linalg.norm(model.w - w_star))
print(f"Loss={loss.item():.5f}, L2|w-w*|={error_norm:.5f} "
      f"Throughput={num_iters/(toc-tic):.1f} it/s")
```

---

## Example workloads

**Important:** 👉 👉 The 2‑D Transformer (Heat2D, Testbed A) and 3‑D PINN (Heat3D, Testbed B) of the paper are released as separate repositories:
- [kbeta-transformer2d](https://github.com/sck-at-ucy/kbeta-transformer2d)
- [kbeta-pinn3d](https://github.com/sck-at-ucy/kbeta-pinn3d)

This repo includes the Transformer – Testbed D (Char-level LM on small-enwik8)

| Folder | Paper section | What it shows | How to run |
|--------|---------------|---------------|------------|
| `examples/transformer_char_lm` | § 6.4 (Testbed D) | Character‑level LM on *small‑enwik8* | `python examples/transformer_char_lm/testbed_d.py --text ./data/small_enwik8.txt --opt kbeta` |


### Running Transformer – Testbed D (Char-level LM on small-enwik8)

All commands assume running from the **repo root** (adjust accordingly)
👉 Make sure you have generated `./data/small-enwik8.txt` and the `./logs_enwi` directory as described below.

Run the Transformer training with the same options used in the paper (adapted to the repo paths):

```bash
  python -u examples/transformer_char_lm/testbed_d.py --text ./data/small-enwik8.txt     --steps 50001 --batch 4 --d_model 512 --n_layer 6 --n_head 8     --ctx 512 --lmin 16 --lmax 512 --warmup 250 --opt kbeta --adam_beta2 0.95     --layer_bucket per-array --barrier_every 100 --eval_every 500     --lr 1e-3     --seed 0 --fixed_eval_seed 1234 --deterministic --compile     --wd 0.0 --lr_schedule "1:1e-3,30000:5e-4,40000:1e-4,60000:1e-5"     2>&1 | tee "logs_enwik/kbeta_seed0.log"
```

This reproduces a run that mirros the testbed reported in the paper with full logging under `logs_enwik/`.

---

### Dataset and creation (verifiable)

We use the **first 30 MB of enwik8** (the classic Hutter Prize corpus).
The slice is created deterministically:

```bash
curl -L -o enwik8.zip https://data.deepai.org/enwik8.zip
unzip enwik8.zip
head -c 30000000 enwik8 > small-enwik8.txt
mkdir -p data && mv small-enwik8.txt data/
mkdir ./logs_enwik
```

Checksums on our machine:

```bash
sha256sum enwik8
# 2b49720e...c024a8

sha256sum data/small-enwik8.txt
# e0152eee...298b7
```

Re-creating `small-enwik8.txt` reproduced the same SHA‑256 (bit‑for‑bit identity).

---

### Model and training protocol

As in the provided script, we train:

* **Architecture**: 6‑block Transformer (`d_model=512`, `n_head=8`, FFN width = 4d)
  GELU, LayerNorm, causal self‑attention; no dropout or weight decay.
* **Data schedule**: variable sequence length with deterministic bucketing
  \(L \in [16,512]\), rounded to multiples of 32; batch = 4; context window = 512.
* **Steps**: 50,001
* **Learning rate schedule**:
  - 1e‑3 for steps 1 ≤ s < 30k
  - 5e‑4 for 30k ≤ s < 40k
  - 1e‑4 for 40k ≤ s ≤ 50k
* **Evaluation**: fixed held‑out batch (length = 256, B = 128) reporting cross‑entropy and BPC.
* **Runs**: 10 matched seeds (0–9).

---

### Optimizers and settings

- **Kourkoutas‑β (ours)**:
  β₁=0.9; dynamic β₂∈[0.88,0.999]; α=0.93 (EMA for sunspike); ε=1e‑8;
  warm‑up=250 steps; `bias_correction="beta2max"`; per‑array stable buckets;
  no AMSGrad/clip/adaptive‑tiny; diagnostics off.

- **Adam‑95**:
  MLX Adam (β₁=0.9, β₂=0.95, ε=1e‑8), bias correction on.

- **Adam‑999**:
  MLX Adam (β₁=0.9, β₂=0.999, ε=1e‑8), bias correction on.

---
## Companion repositories

This repository hosts the **core optimizer implementation** and the **char-level Transformer example** (Testbed D).

Other workloads from the paper are available in dedicated repositories:

- [**kbeta-transformer2d**](https://github.com/sck-at-ucy/kbeta-transformer2d) – 2-D Transformer surrogate for Heat2D (Testbed A).
- [**kbeta-pinn3d**](https://github.com/sck-at-ucy/kbeta-pinn3d) – 3-D Physics-Informed Neural Network for Heat3D (Testbed B).

These companion repos share the same optimizer API and training protocol, so you can directly apply `KourkoutasBeta` with no code changes.

---
## Tests & linting

```bash
pytest                 # unit & ablation tests
ruff check .           # style / imports / naming
pre-commit run --all   # run all hooks (if installed)
```

Continuous Integration (CI) runs these checks automatically.

---

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{Kassinos2025Kourkoutas,
  title   = {Kourkoutas-β: A Sunspike-Driven Adam Optimizer with Desert Flair},
  author  = {Stavros Kassinos},
  journal = {arXiv preprint arXiv:2508.12996},
  year    = {2025},
  url     = {http://arxiv.org/abs/2508.12996}
}
```

---

## License

This work is distributed under the **MIT License**—see [`LICENSE`](LICENSE) for details.

---

## Contributing & roadmap

We welcome issues & PRs!

Planned milestones:

1. **v0.1.0** – optimiser + char‑LM demo (public).
2. **v0.2.0** – PDE workloads migrated to their own repos.
3. **v1.0.0** – journal publication, pip wheels for macOS/Apple Silicon & Linux.

Happy sprinting in the (numerical) desert 🌞🦎🚀📈
