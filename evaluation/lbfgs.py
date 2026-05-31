"""
This file provides two functions:
- `lbfgs`: low-level L-BFGS optimizer using two-loop recursion and a simple backtracking
  line search that attempts to satisfy Armijo + weak curvature condition.
- `reconstruct_lbfgs`: wrapper that builds energy/gradient callables from the inverse
  problem objects (physics, data_fidelity, regularizer) and calls `lbfgs`.

The implementation intentionally uses the provided `f`, `nabla` and `f_and_nabla`
callables so regularizers' precomputed gradient
or energy-aware gradient routines are used when available.
"""

import torch
import numpy as np
from typing import Callable
import inspect


def lbfgs(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nabla: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    f_and_nabla: Callable[[torch.Tensor, torch.Tensor], tuple],
    max_iter: int = 200,
    m: int = 10,
    tol: float = 1e-4,
    max_ls_iter: int = 40,
    c1: float = 1e-4,
    c2: float = 0.9,
    verbose: bool = False,
):
    """
    L-BFGS optimizer (per-sample) using the standard two-loop recursion.

    Returns: (rec, None, steps, converged)
    - `rec`: reconstructed tensor with same shape as `x0`
    - `steps`: number of iterations performed (per-sample max)
    - `converged`: boolean tensor (B,) indicating which samples converged
    """

    device = x0.device
    dtype = x0.dtype

    batch = x0.shape[0]
    rec = x0.clone().detach()
    converged = torch.zeros(batch, dtype=torch.bool, device=device)
    steps_per_sample = torch.zeros(batch, dtype=torch.int32, device=device)

    # Work on each sample independently (safe and simple)
    for b in range(batch):
        xb = x0[b : b + 1].detach().clone()
        yb = y[b : b + 1]

        # initial energy and gradient
        try:
            energy_val, grad = f_and_nabla(xb, yb)
        except Exception:
            energy_val = f(xb, yb)
            grad = nabla(xb, yb)

        energy_val = energy_val.reshape(-1)[0]
        g = grad.reshape(-1).detach().clone()

        # history lists (store flattened tensors)
        s_list = []
        y_list = []

        # main loop
        for k in range(max_iter):
            # Two-loop recursion to compute search direction
            q = g.clone()
            alpha_list = []
            rho_list = []
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                rho_i = 1.0 / (torch.dot(yi, si) + 1e-12)
                alpha_i = rho_i * torch.dot(si, q)
                q = q - alpha_i * yi
                alpha_list.append(alpha_i)
                rho_list.append(rho_i)

            # initial H0 as scalar: (s_{k-1}^T y_{k-1})/(y_{k-1}^T y_{k-1})
            if len(s_list) > 0:
                s_last = s_list[-1]
                y_last = y_list[-1]
                denom = torch.dot(y_last, y_last)
                H0 = (torch.dot(s_last, y_last) / (denom + 1e-12)).item()
                if not np.isfinite(H0) or H0 <= 0:
                    H0 = 1.0
            else:
                H0 = 1.0
            r = q * H0

            # second loop
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho_i = rho_list[len(s_list) - 1 - i]
                alpha_i = alpha_list[len(s_list) - 1 - i]
                beta = rho_i * torch.dot(yi, r)
                r = r + si * (alpha_i - beta)

            p = -r  # search direction (flattened)

            # ensure descent direction
            g_dot_p = torch.dot(g, p)
            if g_dot_p >= 0:
                # fallback to negative gradient
                p = -g
                g_dot_p = torch.dot(g, p)

            # line search (backtracking with Armijo + weak curvature check)
            alpha = 1.0
            accepted = False
            for ls in range(max_ls_iter):
                xb_new = (xb.reshape(-1) + alpha * p).view_as(xb)
                try:
                    f_new = f(xb_new, yb).reshape(-1)[0]
                except Exception:
                    f_new = f_and_nabla(xb_new, yb)[0].reshape(-1)[0]

                if f_new <= energy_val + c1 * alpha * g_dot_p:
                    # compute new gradient (needed for curvature check)
                    grad_new = nabla(xb_new, yb)
                    g_new = grad_new.reshape(-1).detach().clone()
                    if torch.dot(g_new, p) >= c2 * g_dot_p:
                        accepted = True
                        break
                alpha *= 0.5

            if not accepted:
                # accept last trial anyway (stabilize)
                xb_new = (xb.reshape(-1) + alpha * p).view_as(xb)
                try:
                    grad_new = nabla(xb_new, yb)
                    g_new = grad_new.reshape(-1).detach().clone()
                    f_new = f(xb_new, yb).reshape(-1)[0]
                except Exception:
                    f_new, grad_new = f_and_nabla(xb_new, yb)
                    f_new = f_new.reshape(-1)[0]
                    g_new = grad_new.reshape(-1).detach().clone()

            s = (xb_new.reshape(-1) - xb.reshape(-1)).detach().clone()
            yk = (g_new - g).detach().clone()

            # update memory
            if torch.dot(s, yk) > 1e-12:
                if len(s_list) == m:
                    s_list.pop(0)
                    y_list.pop(0)
                s_list.append(s)
                y_list.append(yk)

            # update iterate
            xb = xb_new.detach().clone()
            g = g_new
            energy_val = f_new

            # check convergence (relative step)
            step_norm = torch.norm(s)
            x_norm = torch.norm(xb.reshape(-1))
            if step_norm / (x_norm + 1e-12) < tol:
                converged[b] = True
                steps_per_sample[b] = k + 1
                break

            # if last iter reached mark steps
            if k == max_iter - 1:
                steps_per_sample[b] = max_iter

        rec[b : b + 1] = xb.detach()

    # Return rec, L, steps, converged
    # We don't estimate a Lipschitz constant here, so return L as a zero-tensor
    L_placeholder = torch.tensor(0.0, dtype=torch.float32, device=device)
    # steps: return the maximum steps used across the batch
    steps = int(steps_per_sample.max().item())
    return rec, L_placeholder, steps, converged


def reconstruct_lbfgs(
    y,
    physics,
    data_fidelity,
    regularizer,
    lamda,
    max_iter,
    tol,
    x_init=None,
    detach_grads=True,
    verbose=False,
    return_stats=False,
):
    """
    The wrapper creates `energy`, `energy_grad` and `energy_and_grad` callables and then
    calls `lbfgs`. The wrapper returns the reconstruction and (optionally) a stats dict
    containing keys `L`, `steps`, and `converged` to be compatible with `evaluate()`.
    """

    if x_init is not None:
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)

    def energy(val, y_in):
        with torch.no_grad():
            fun = data_fidelity(val, y_in, physics) + lamda * regularizer.g(val)
        if detach_grads:
            fun = fun.detach()
        return fun.reshape(-1)

    def energy_grad(val, y_in):
        grad = data_fidelity.grad(val, y_in, physics) + lamda * regularizer.grad(val)
        if detach_grads:
            grad = grad.detach()
        return grad

    # check if regularizer.grad supports providing energy as well
    signature = inspect.signature(regularizer.grad)
    argument_names = [param.name for param in signature.parameters.values()]
    if "get_energy" in argument_names:

        def energy_and_grad(val, y_in):
            fun, grad = regularizer.grad(val, get_energy=True)
            fun = data_fidelity(val, y_in, physics) + lamda * fun
            grad = data_fidelity.grad(val, y_in, physics) + lamda * grad
            if detach_grads:
                fun = fun.detach()
                grad = grad.detach()
            return fun.reshape(-1), grad

    else:
        energy_and_grad = lambda val, y_in: (energy(val, y_in), energy_grad(val, y_in))

    rec, L, steps, converged = lbfgs(
        x0=x,
        y=y,
        f=energy,
        nabla=energy_grad,
        f_and_nabla=energy_and_grad,
        max_iter=max_iter,
        m=10,
        tol=tol,
        verbose=verbose,
    )

    stats = dict(L=L.detach() if isinstance(L, torch.Tensor) else torch.tensor(L), steps=steps, converged=converged)
    if return_stats:
        return rec, stats
    return rec
