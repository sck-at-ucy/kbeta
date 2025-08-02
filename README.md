[![CI (macOS arm64)](https://github.com/sck-at-ucy/kbeta/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sck-at-ucy/kbeta/actions/workflows/ci.yml)

# kbeta – *Kourkoutas‑β Optimiser*   🌞🦎🚀📈

> **Research code for our upcoming paper
> “Kourkoutas‑β: Soft‑max Momentum with Adaptive Variance for Mesh‑Accelerated Deep Learning.”**
> The repository ships the optimiser **plus two demonstration workloads** (a 2‑D data‑driven Transformer and a 3‑D PINN).

---

## Table of Contents
1. [Key ideas](#key-ideas)
2. [Project layout](#project-layout)
3. [Quick start](#quick-start)
4. [Using Kourkoutas‑β in your own model](#minimal-example)
5. [Running the demo workloads](#demo-workloads)
6. [Tests & linting](#tests--linting)
7. [Citation](#citation)
8. [License](#license)
9. [Contributing & roadmap](#contributing--roadmap)

---

## Key ideas

* **Soft‑max variance tracking** to tame gradient spikes.
* **Two β₂ parameters**:
  *β₂_min* for ultra‑fast warm‑up, *β₂_max* for long‑term stability.
* **Layer‑wise adaptive tiny‑values** (ϵ, spike dampers) that shrink with training progress.
* 100 % **Apple‑MLX** compatible – no PyTorch required.

See detailed derivations in the forthcoming pre‑print (link will appear here).

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
├── src/kbeta/               # pip package
│   ├── __init__.py          # re‑exports optimiser
│   └── optim/
│       ├── __init__.py
│       └── kbeta_softmax.py # <-- KourkoutasSoftmaxFlex implementation
│
├── workloads/
│   ├── transformer/         # 2‑D heat‑diffusion demo
│   └── pinn3d/              # 3‑D PINN demo
│
├── tests/                   # pytest suite (incl. smoke test)
├── docs/                    # sphinx material (optional)
├── pyproject.toml
└── README.md                # you are here
```

---

## Quick start

```bash
# 1. clone *your* fork (recommended) and cd into it
git clone git@github.com:<YOUR-USERNAME>/kbeta.git
cd kbeta

# 2. create a fresh virtualenv
python -m venv .venv && source .venv/bin/activate

# 3. editable install + dev extras
pip install -e ".[dev]"

# 4. run the ultra‑short smoke test
pytest -q                       # should print ‘1 passed’
```

---

## Minimal example

```python
import mlx.core as mx
from kbeta.optim import KourkoutasSoftmaxFlex
import mlx.nn as nn

# dummy single‑parameter model
class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.zeros((3,))

    def __call__(self, x):
        return (self.w * x).sum()

model = Dummy()
optim = KourkoutasSoftmaxFlex(learning_rate=1e-3)
optim.init(model.parameters())

x = mx.ones((3,))
loss, grads = nn.value_and_grad(model)(model, x)
optim.update(model, grads)  # ← one training step
```

---

## Demo workloads

| Folder | Paper section | What it shows | How to run |
|--------|---------------|---------------|------------|
| `workloads/transformer` | § 4.1 | 2‑D heat‑diffusion **data‑driven Transformer** trained with Kourkoutas‑β vs Adam | `python -m transformer.Train --config configs/base.yaml` |
| `workloads/pinn3d` | § 4.2 | 3‑D physics‑informed neural network (**PINN**) on a diffusion PDE | `python train_pinn3d.py --optimizer kourkoutas` |

All configs are pure YAML; command‑line `--override KEY=VAL` flags allow rapid sweeps.

---

## Tests & linting

```bash
pytest                 # unit tests
ruff check .           # style / import / naming
pre-commit run --all   # everything (if you installed the hooks)
```

Continuous Integration (CI) will refuse a PR that fails any of the above.

---

## Citation

```
@article{Kourkoutas2025,
  title   = {Kourkoutas‑β: Soft‑max Momentum with Adaptive Variance},
  author  = {S. Kassinos and et al.},
  journal = {ArXiv preprint},
  year    = {2025},
  url     = {https://arxiv.org/abs/XXXXX}
}
```

---

## License

This work is distributed under the **MIT License**—see [`LICENSE`](LICENSE) for details.

---

## Contributing & roadmap

We welcome issues & PRs!
Planned milestones:

1. **v0.1.0** – optimiser + 2‑D Transformer demo (public).
2. **v0.2.0** – 3‑D PINN demo, mixed‑precision benchmarks.
3. **v1.0.0** – journal paper release, pip wheels for macOS/Apple Silicon & Linux.

If you run into trouble, open an issue or ping `@stavros‑k` on GitHub.

Happy sprinting in the (numerical) desert 🌞🦎🚀📈
