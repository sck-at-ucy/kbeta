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
4. [Using Kourkoutas‑β in your own model](#minimal-example)
5. [Example workloads](#example-workloads)
6. [Tests & linting](#tests--linting)
7. [Citation](#citation)
8. [License](#license)
9. [Contributing & roadmap](#contributing--roadmap)

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
├── assets/                      # logo and figures
├── pyproject.toml
└── README.md                    # you are here
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

## Minimal example

```import time
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
print(f"Loss={loss.item():.5f}, L2|w-w*|={error_norm:.5f}, Throughput={num_iters/(toc-tic):.1f} it/s")
```

---

## Example workloads

| Folder | Paper section | What it shows | How to run |
|--------|---------------|---------------|------------|
| `examples/transformer_char_lm` | § 6.4 (Testbed D) | Character‑level LM on *small‑enwik8* | `python examples/transformer_char_lm/testbed_d.py --text ./data/small_enwik8.txt --opt kbeta` |

The 2‑D Transformer (Heat2D, Testbed A) and 3‑D PINN (Heat3D, Testbed B) are released as separate repositories:
- [kbeta-transformer2d](https://github.com/sck-at-ucy/kbeta-transformer2d)
- [kbeta-pinn3d](https://github.com/sck-at-ucy/kbeta-pinn3d)

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

```
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
