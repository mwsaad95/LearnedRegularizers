# Derived from pytorch-minimize (MIT, (c) 2021 Reuben Feinman); see LICENSE.
from collections import deque
import torch
from .line_search import strong_wolfe, weak_wolfe
import matplotlib.pyplot as plt


@torch.no_grad()
def lbfgs(
    fun_and_grad,
    x0,
    history_size=10,
    max_iter=200,
    tol=1e-6,
    gtol=1e-5,
    gtd_tol=1e-10,
    verbose=False,
    c1=1e-4,
    c2=0.9,
    tolerance_change=1e-6,
    max_ls=25,
    line_search_variant="strong",
    damping_eps=None,
):
    """L-BFGS minimizer (Nocedal & Wright §7.4). See ``reconstruct`` for parameter docs."""
    if damping_eps is None:
        damping_eps = None if line_search_variant == "strong" else 0.2

    if line_search_variant not in ("strong", "weak"):
        raise ValueError(
            f"Invalid line_search_variant: {line_search_variant}. "
            "Must be 'strong' or 'weak'."
        )
    ls_func = strong_wolfe if line_search_variant == "strong" else weak_wolfe
    x_shape = x0.shape
    nfev = 0

    def _eval(x_flat):
        nonlocal nfev
        nfev += 1
        f, g = fun_and_grad(x_flat.reshape(x_shape))
        return f.squeeze(), g.flatten()

    def _dir_eval(x, t, d):
        return _eval(x + d.mul(t))

    x = x0.flatten().clone()
    f, g = _eval(x)
    g_norm_0 = g.norm().clamp(min=1e-12)
    if verbose:
        print("initial fval: %0.4f" % f)

    # Inverse-Hessian state for the L-BFGS two-loop recursion.
    history = deque(maxlen=history_size)
    H_diag = 1.0
    d = g.neg()
    t = min(1.0, 1.0 / g.abs().sum())
    n_iter = 0
    converged = False

    # 1. Initialize empty lists to store historical metrics
    f_history = []
    grad_norm_history = []
    rel_step_history = []

    for n_iter in range(1, max_iter + 1):

        # --- Quasi-Newton direction (two-loop recursion: d = -H g) ---
        if n_iter > 1:
            d = g.neg()
            alphas = []
            for s_i, y_i, rho_i in reversed(history):
                a = rho_i * s_i.dot(d)
                alphas.append(a)
                d.addcmul_(y_i, a, value=-1.0)  # d -= a * y_i (in-place, no sync)
            d.mul_(H_diag)
            for (s_i, y_i, rho_i), a in zip(history, reversed(alphas)):
                beta_i = rho_i * y_i.dot(d)
                d.addcmul_(s_i, a - beta_i)  # d += (a - beta_i) * s_i

        gtd = g.dot(d)  # directional derivative; must be negative for descent
        if gtd > -gtd_tol:
            if verbose:
                print("A non-descent direction was encountered.")
            break

        # --- line search ---
        f_new, g_new, t = ls_func(
            _dir_eval,
            x,
            t,
            d,
            f,
            g,
            gtd,
            c1=c1,
            c2=c2,
            tolerance_change=tolerance_change,
            max_ls=max_ls,
        )

        # --- Hessian update: store curvature pair (s, y) ---
        s = d.mul(t)
        y = g_new.sub(g)
        rho_inv = y.dot(s)
        if damping_eps is not None:
            # Powell damping (Nocedal & Wright §18.3): when y^T s is too small
            # relative to s^T B s, blend y toward Bs = -t g (exact because the
            # L-BFGS direction satisfies B d = -g, so B s = B(t d) = -t g) so the
            # curvature condition holds instead of skipping the pair.
            Bs = g.mul(-t)
            sBs = s.dot(Bs)
            if rho_inv < damping_eps * sBs:
                theta = (1.0 - damping_eps) * sBs / (sBs - rho_inv)
                y = theta * y + (1.0 - theta) * Bs
                rho_inv = y.dot(s)
        if rho_inv > 1e-10:
            history.append((s, y, 1.0 / rho_inv))
            H_diag = rho_inv / y.dot(y)

        f = f_new
        x.add_(s)
        g = g_new
        t = 1.0

        # 2. Append values into history tracking lists at the end of each iteration
        f_history.append(f.item() if hasattr(f, 'item') else f)
        grad_norm_history.append(g.norm().item())
        rel_step_history.append((s.norm() / x.norm().clamp(min=1e-12)).item())

        if s.norm() / x.norm().clamp(min=1e-12) <= tol:
            converged = True
            break

        if g.norm() / g_norm_0 <= gtol:
            converged = True
            break

        if not f.isfinite():
            if verbose:
                print("Precision loss; desired accuracy not achieved.")
            break

    else:
        if verbose:
            print("Maximum number of iterations exceeded.")

    if verbose:
        print("Current function value: %f" % f)
        print("Iterations: %d" % n_iter)
        print("Function evaluations: %d" % nfev)

    # =======================================================
    # 3. Direct Plotting Code: Generate and save individual 2D plots
    # =======================================================
    
    # Plot 1: Objective Function Value History (Just F)
    plt.figure(figsize=(7, 5))
    plt.plot(f_history, color='blue', linewidth=2, label='Function Value (F)')
    plt.xlabel('Iterations (iter.)')
    plt.ylabel('Function Value (F)')
    plt.title('Convergence History - Function Value')
    plt.grid(True)
    plt.legend()
    plt.savefig('plot_1_function_value.png', dpi=300)
    plt.close()

    # Plot 2: Gradient Norm History (||∇F||)
    plt.figure(figsize=(7, 5))
    plt.plot(grad_norm_history, color='cyan', linestyle='--', linewidth=2, label='Gradient Norm ||∇F||')
    plt.xlabel('Iterations (iter.)')
    plt.ylabel('Gradient Norm ||∇F||')
    plt.title('Convergence History - Gradient Norm')
    plt.grid(True)
    plt.legend()
    plt.savefig('plot_2_gradient_norm.png', dpi=300)
    plt.close()

    # Plot 3: Relative Step Size vs Tolerance Threshold Line
    plt.figure(figsize=(7, 5))
    plt.plot(rel_step_history, color='darkblue', linewidth=2, label='Relative Step Size')
    plt.axhline(y=tol, color='red', linestyle=':', label=f'Tolerance (tol={tol})')
    plt.yscale('log')
    plt.xlabel('Iterations (iter.)')
    plt.ylabel('||x_k - x_{k-1}|| / ||x_k||')
    plt.title('Relative Step Size vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_3_relative_step_size.png', dpi=300)
    plt.close()
    
    return x.view_as(x0), n_iter, converged