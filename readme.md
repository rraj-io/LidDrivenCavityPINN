# Lid-Driven Cavity — Physics-Informed Neural Network

A PyTorch implementation of a Physics-Informed Neural Network (PINN) that solves the 2D unsteady incompressible Navier–Stokes equations for the classical **lid-driven cavity** benchmark at **Re = 100**.

The network is trained purely from the PDE residuals and boundary conditions — no CFD reference solution is fed in as data — and is validated against the canonical Ghia, Ghia & Shin (1982) results.

---

## Problem setup

A unit square cavity $\Omega = [0,1] \times [0,1]$ with a lid that slides at $u = 1$.

- **Domain:** $(x, y) \in [0, 1]^2$
- **Time horizon:** $t \in [0, 5]$
- **Reynolds number:** $\mathrm{Re} = 100 \;\Rightarrow\; \nu = 1/\mathrm{Re} = 0.01$

**Boundary conditions**
- Top lid ($y = 1$): $u = 1,\; v = 0$
- Left, right, bottom walls: $u = 0,\; v = 0$ (no-slip)

**Governing equations** — incompressible Navier–Stokes in residual form:

$$
\begin{aligned}
f_x &= u_t + u\,u_x + v\,u_y + p_x - \nu\,(u_{xx} + u_{yy}) = 0 \\
f_y &= v_t + u\,v_x + v\,v_y + p_y - \nu\,(v_{xx} + v_{yy}) = 0
\end{aligned}
$$

Continuity ($\nabla \cdot \mathbf{u} = 0$) is enforced **structurally** by representing the velocity through a stream function $\psi$:

$$
u = \frac{\partial \psi}{\partial y}, \qquad v = -\frac{\partial \psi}{\partial x}
$$

so $\nabla \cdot \mathbf{u} = 0$ holds automatically by construction.

---

## Network architecture

A fully connected MLP $\mathcal{N}_\theta : (x, y, t) \mapsto (p, \psi)$ — pressure and stream function — from which $(u, v)$ and all PDE residuals are obtained by automatic differentiation.

| Component | Value |
|---|---|
| Input | $(x, y, t)$ — 3 features |
| Output | $(p, \psi)$ — 2 features |
| Hidden width | 20 |
| Hidden depth | 8 residual-style linear blocks + a 3→64→20 input stem |
| Activation | SiLU |
| Optimizer | Adam, learning rate $5 \times 10^{-3}$ |
| Epochs | 1000 |

## Loss function

A weighted sum of four collocation-point losses, evaluated on freshly resampled points each epoch:

| Term | Where it is enforced | What it enforces |
|---|---|---|
| Initial condition | $N_{\text{init}} = 64$ points at $t = 0$ | $u = v = 0$ and PDE residuals |
| No-slip walls | $N_{\text{boundary}} = 16$ × 3 sides | $u = v = 0$ |
| Moving lid | $N_{\text{upper}} = 64$ | $u = 1,\; v = 0$ |
| Interior PDE residual | $N_{\text{mesh}} = 128$ | $f_x = f_y = 0$ (weighted ×2) |

---

## Repository structure

```
LidDrivenCavityPINN/
├── main.py        # Training loop, plotting, Ghia comparison
├── pinn.py        # LidDrivenCavityPINN class — network + residual computation
├── initial.py     # Random collocation point samplers (IC, BC, domain)
├── utils.py       # Seed setting and device selection
├── cavity.pt      # Trained model weights (saved after run)
├── cavity.pth     # Checkpoint (loaded on resume if present)
└── results/       # Output figures from a trained run
```

---

## Running

```bash
pip install torch numpy matplotlib tqdm
python main.py
```

Training resumes from `cavity.pth` if present; otherwise it starts from scratch. Final weights are written to `cavity.pt`. On a modern laptop CPU the 1000-epoch run finishes in a few minutes; with CUDA it's a matter of seconds.

---

## Results

All plots are evaluated at $t = 0.8\,T = 4.0$ on a 50 × 50 grid.

### Centerline velocity vs. Ghia et al. (1982)

Vertical profile of the horizontal velocity $u(x = 0.5, y)$ compared against the benchmark data from Ghia, Ghia & Shin (1982).

![Velocity comparison](results/velocity_comparison.png)

**What the plot shows.** The PINN reproduces the upper-cavity shear layer faithfully — the steep gradient near the lid and the boundary values at $y = 1$ and $y = 0.85$ are essentially on top of the benchmark. Below $y \approx 0.7$ the network predicts $u \approx 0$, while Ghia's reference shows a clear negative bulge with a minimum of $u \approx -0.21$ near $y \approx 0.45$ — the return flow of the primary vortex.

**Interpretation.** The model has learned the driven shear layer but has not yet captured the closed primary circulation. This is a well-known failure mode of vanilla PINNs on cavity flow at moderate Reynolds numbers — see *Future work* below for the standard remedies (curriculum on Re, denser interior collocation, longer training, second-order optimizer).

### Velocity field $(u, v)$

![Velocity field](results/velocity_field.png)

A dense quiver of the predicted velocity. Strong horizontal flow near the lid, decaying with depth — consistent with the centerline profile above.

### Stream function $\psi$

![Stream function](results/stream_function.png)

$\psi$ is stratified along $y$ rather than forming a closed circulating eddy, mirroring the missing return flow seen in the centerline comparison.

### Pressure field $p$

![Pressure field](results/pressure_field.png)

Pressure varies on the order of $\sim 0.01$ across the domain, with the low-pressure region beneath the lid — physically sensible, since pressure in incompressible flow is determined only up to a constant.

### Momentum residuals $f_x$, $f_y$

These are the residuals of the momentum equations evaluated on the prediction; they should be zero everywhere a perfect solver would converge.

| $f_x$ | $f_y$ |
|---|---|
| ![fx residual](results/fx_field.png) | ![fy residual](results/fy_field.png) |

Residuals are at most $\mathcal{O}(10^{-2})$ and concentrated in a thin strip just below the lid where the shear layer is sharpest. Elsewhere they are $\sim 10^{-3}$ or below.

### Centerline profile (PINN only)

![Velocity profile](results/velocity_profile.png)

The same $u(0.5, y)$ profile without the benchmark overlay — useful to inspect the network's own prediction in isolation.

---

## Known limitations

- The recirculation in the lower half of the cavity is not captured at the current training budget. The PINN converges to a solution that satisfies the boundary conditions and momentum equations on the sampled collocation points without recovering the global vortex structure — a common pathology that gets worse with Re.
- Adam alone tends to stall on PINN problems; an LBFGS refinement stage (already stubbed in `pinn.py`) is the usual fix.
- Collocation point counts ($N_{\text{mesh}} = 128$ interior, 64 lid, 16 per wall) are small. Cavity benchmarks in the literature typically use $10^3$–$10^4$ interior points.

## Future work

- **Two-stage optimization:** Adam to a reasonable region, then LBFGS to drive residuals down further.
- **Adaptive / hard-mining sampling** of collocation points in regions of high residual (RAR, RAD).
- **Curriculum on Re**, training first at Re = 10 and warm-starting up to Re = 100, 400, 1000.
- **Loss balancing** between BC, IC, and PDE terms — manual weights here are uniform, which often under-emphasizes the PDE residual.
- Sweep architectures (depth, width, Fourier features) and compare against the Ghia benchmark quantitatively (e.g., $L_2$ error on the centerline).

---

## References

1. Ghia, U., Ghia, K. N., & Shin, C. T. (1982). *High-Re solutions for incompressible flow using the Navier–Stokes equations and a multigrid method.* Journal of Computational Physics, 48(3), 387–411.
2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.

## Acknowledgement

PINN structure adapted from [ehwan/PINN](https://github.com/ehwan/PINN/tree/main/cavity).
