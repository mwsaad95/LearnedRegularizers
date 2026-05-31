"""
This file provides two functions:
- `newton_cg`: low-level Newton-CG optimizer using an inner conjugate-gradient
  solve for the Newton system and a backtracking line-search.
- `reconstruct_newton_cg`: wrapper that builds `energy`, `energy_grad` and
  `energy_and_grad` callables from `physics`, `data_fidelity` and `regularizer`
  and calls `newton_cg`.

The implementation uses finite-difference Hessian-vector products by default
which works with analytic gradient routines provided by `regularizer.grad`
and `data_fidelity.grad` (no autograd graph is required).
"""

import torch
from typing import Callable
import inspect


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1)


def _unflat(flat: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    return flat.view_as(template)


def _hvp_fd(xb: torch.Tensor, v_flat: torch.Tensor, nabla: Callable, yb: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Finite-difference Hessian-vector product: (grad(x+eps*v)-grad(x))/eps.

    xb: tensor shaped as input (batch size 1)
    v_flat: flattened vector
    nabla: callable nabla(val, y) returning gradient with same shape as val
    yb: corresponding y (batch size 1)
    Returns flattened hvp tensor
    """
    grad0 = _flat(nabla(xb, yb)).detach()
    xb_pert = (_unflat(_flat(xb) + eps * v_flat, xb)).detach()
    grad1 = _flat(nabla(xb_pert, yb)).detach()
    return (grad1 - grad0) / eps


def _cg_solve(Avp, b, x0=None, tol=1e-4, max_iter=50):
    """Conjugate gradient solver for symmetric positive-definite linear systems.

    Solves A x = b where Avp(v) = A @ v.
    All inputs are 1D flattened torch tensors on the same device.
    """
    device = b.device
    if x0 is None:
        x = torch.zeros_like(b, device=device)
    else:
        x = x0.clone().to(device)
    r = b - Avp(x)
    p = r.clone()
    rsold = torch.dot(r, r)
    if rsold.sqrt() < tol:
        return x, 0
    for i in range(max_iter):
        Ap = Avp(p)
        denom = torch.dot(p, Ap)
        if denom.abs() < 1e-20:
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            return x, i + 1
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, i + 1


def newton_cg(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nabla: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    f_and_nabla: Callable[[torch.Tensor, torch.Tensor], tuple],
    max_iter: int = 5,
    cg_tol: float = 1e-3,
    cg_max_iter: int = 50,
    tol: float = 1e-4,
    max_ls_iter: int = 20,
    c1: float = 1e-4,
    verbose: bool = False,
):
    """
    Newton-CG optimizer.
    Operates per-sample (batch dimension supported). Returns (rec, L_placeholder, steps, converged)
    """
    device = x0.device
    batch = x0.shape[0]
    rec = x0.clone().detach()
    converged = torch.zeros(batch, dtype=torch.bool, device=device)
    steps_per_sample = torch.zeros(batch, dtype=torch.int32, device=device)

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
        g = _flat(grad).detach().clone()

        for k in range(max_iter):
            g_norm = torch.norm(g)
            if g_norm.item() < tol:
                converged[b] = True
                steps_per_sample[b] = k
                break

            # solve (approximately) H p = -g using conjugate gradient where H v ~= hvp_fd
            b_cg = -g

            def Avp(v_flat: torch.Tensor) -> torch.Tensor:
                return _hvp_fd(xb, v_flat, nabla, yb)

            p_flat, cg_iters = _cg_solve(Avp, b_cg, tol=max(cg_tol * torch.norm(b_cg).item(), 1e-12), max_iter=cg_max_iter)

            # line search along p
            # ensure descent
            if torch.dot(g, p_flat) >= 0:
                # not a descent direction; fallback to negative gradient
                p_flat = -g

            alpha = 1.0
            accepted = False
            for ls in range(max_ls_iter):
                xb_new = _unflat(_flat(xb) + alpha * p_flat, xb)
                try:
                    f_new = f(xb_new, yb).reshape(-1)[0]
                except Exception:
                    f_new = f_and_nabla(xb_new, yb)[0].reshape(-1)[0]

                if f_new <= energy_val + c1 * alpha * torch.dot(g, p_flat):
                    # accept
                    try:
                        _, grad_new = f_and_nabla(xb_new, yb)
                        g_new = _flat(grad_new).detach().clone()
                    except Exception:
                        g_new = _flat(nabla(xb_new, yb)).detach().clone()
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                # accept last trial
                xb_new = _unflat(_flat(xb) + alpha * p_flat, xb)
                try:
                    _, grad_new = f_and_nabla(xb_new, yb)
                    g_new = _flat(grad_new).detach().clone()
                    f_new = f(xb_new, yb).reshape(-1)[0]
                except Exception:
                    f_new = f(xb_new, yb).reshape(-1)[0]
                    g_new = _flat(nabla(xb_new, yb)).detach().clone()

            s = ( _flat(xb_new) - _flat(xb) ).detach().clone()

            xb = xb_new.detach().clone()
            g = g_new
            energy_val = f_new

            step_norm = torch.norm(s)
            x_norm = torch.norm(_flat(xb))
            if step_norm / (x_norm + 1e-12) < tol:
                converged[b] = True
                steps_per_sample[b] = k + 1
                break

            if k == max_iter - 1:
                steps_per_sample[b] = max_iter

        rec[b : b + 1] = xb

    L_placeholder = torch.tensor(0.0, dtype=torch.float32, device=device)
    steps = int(steps_per_sample.max().item())
    return rec, L_placeholder, steps, converged


def reconstruct_newton_cg(
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
    Builds `energy`, `energy_grad` and `energy_and_grad` callables and then
    calls `newton_cg`. Returns reconstruction and (optionally) stats dict
    with keys `L`, `steps`, `converged` to be compatible with `evaluate()`.
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

    rec, L, steps, converged = newton_cg(
        x0=x,
        y=y,
        f=energy,
        nabla=energy_grad,
        f_and_nabla=energy_and_grad,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    stats = dict(L=L.detach() if isinstance(L, torch.Tensor) else torch.tensor(L), steps=steps, converged=converged)
    if return_stats:
        return rec, stats
    return rec
