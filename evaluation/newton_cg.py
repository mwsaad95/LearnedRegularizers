import torch
from torchmin import minimize

def newton_cg(x0, energy_func, max_iter=200, tol=1e-4, verbose=False):
    """
    Algorithm: Newton-CG optimization using pytorch-minimize
    This implements the line search as requested.
    """
    if verbose:
        print("Starting Newton-CG optimization...")
        
    # Using the minimize function from torchmin
    result = minimize(
        energy_func,           # The objective function (energy)
        x0,                    # Initial guess
        method='newton-cg',       # Algorithm name
        max_iter=max_iter,     # Maximum iterations
        tol=tol,               # Stopping tolerance
        options=dict(line_search='strong-wolfe'), # The linesearch requested by the supervisor
        disp=1 if verbose else 0
    )
    
    # Return the final image, number of iterations, and success status
    return result.x, result.nit, result.success


def reconstruct_newton_cg(
    y, physics, data_fidelity, regularizer, lamda, max_iter, tol, x_init=None, verbose=False, return_stats=False
):
    """Wrapper to use Newton-CG for the inverse problem"""
    
    # 1. Initialization
    if x_init is not None:
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)
        
    # We must require gradients for the Newton-CG optimizer to work
    x.requires_grad_(True)

    # 2. Define the energy function
    def energy_for_torchmin(val):
        # Calculate data fidelity and regularization
        fun = data_fidelity(val, y, physics) + lamda * regularizer.g(val)
        # torchmin requires a single scalar output
        return fun.sum()

    # 3. Run the solver
    rec, steps, converged = newton_cg(
        x0=x,
        energy_func=energy_for_torchmin,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose
    )
    
    # Detach the result so it doesn't carry gradient history unnecessarily
    rec = rec.detach()

    # 4. Return results and statistics
    stats = dict(steps=steps, converged=converged)
    if return_stats:
        return rec, stats
    return rec
