"""
Finite Difference Method (FDM) Solver for PDE-based Glaucoma Modeling

This module implements finite difference methods for solving:
1. 1D Poisson equation (validation)
2. 2D Poisson equation (steady-state pressure)
3. 2D Navier-Stokes equations (aqueous humor flow)
"""

import numpy as np
from scipy.sparse import diags, kron, eye, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class FDMSolver1D:
    """
    Finite Difference Method solver for 1D problems.

    Solves: -u'' = f(x) on [a, b]
    with Dirichlet boundary conditions: u(a) = ua, u(b) = ub
    """

    def __init__(self, a=0.0, b=1.0, n=100):
        """
        Initialize 1D FDM solver.

        Parameters:
        -----------
        a : float
            Left boundary
        b : float
            Right boundary
        n : int
            Number of interior grid points
        """
        self.a = a
        self.b = b
        self.n = n
        self.h = (b - a) / (n + 1)  # Grid spacing
        self.x = np.linspace(a, b, n + 2)  # Include boundary points

    def solve_poisson(self, f, ua=0.0, ub=0.0):
        """
        Solve 1D Poisson equation: -u'' = f(x)

        Parameters:
        -----------
        f : callable
            Right-hand side function f(x)
        ua, ub : float
            Boundary values at x=a and x=b

        Returns:
        --------
        u : ndarray
            Solution array (including boundary points)
        """
        # Build stiffness matrix A (tridiagonal)
        # -u'' ≈ (-u_{i-1} + 2u_i - u_{i+1}) / h^2
        diag_vals = [
            -np.ones(self.n - 1),  # Lower diagonal
            2 * np.ones(self.n),   # Main diagonal
            -np.ones(self.n - 1)   # Upper diagonal
        ]
        A = diags(diag_vals, [-1, 0, 1], format='csr') / (self.h**2)

        # Build right-hand side vector
        f_vec = f(self.x[1:-1])  # Evaluate f at interior points

        # Apply boundary conditions
        f_vec[0] += ua / (self.h**2)
        f_vec[-1] += ub / (self.h**2)

        # Solve linear system
        u_interior = spsolve(A, f_vec)

        # Combine with boundary values
        u = np.zeros(self.n + 2)
        u[0] = ua
        u[1:-1] = u_interior
        u[-1] = ub

        return u

    def convergence_study(self, f, u_exact, ua=0.0, ub=0.0, n_values=None):
        """
        Perform mesh refinement study to demonstrate convergence.

        Parameters:
        -----------
        f : callable
            Right-hand side function
        u_exact : callable
            Exact solution (for error computation)
        ua, ub : float
            Boundary values
        n_values : list of int
            Mesh sizes to test (default: [10, 20, 40, 80, 160])

        Returns:
        --------
        h_vals : ndarray
            Mesh spacings
        errors : ndarray
            L2 errors
        """
        if n_values is None:
            n_values = [10, 20, 40, 80, 160]

        h_vals = []
        errors = []

        for n in n_values:
            solver = FDMSolver1D(self.a, self.b, n)
            u_h = solver.solve_poisson(f, ua, ub)
            u_ex = u_exact(solver.x)

            # Compute L2 error
            error = np.sqrt(solver.h * np.sum((u_h - u_ex)**2))

            h_vals.append(solver.h)
            errors.append(error)

        return np.array(h_vals), np.array(errors)


class FDMSolver2D:
    """
    Finite Difference Method solver for 2D problems.

    Solves: -Δu = f(x, y) on rectangular domain [ax, bx] × [ay, by]
    with Dirichlet boundary conditions
    """

    def __init__(self, ax=0.0, bx=1.0, ay=0.0, by=1.0, nx=50, ny=50):
        """
        Initialize 2D FDM solver.

        Parameters:
        -----------
        ax, bx : float
            Domain bounds in x-direction
        ay, by : float
            Domain bounds in y-direction
        nx, ny : int
            Number of interior grid points in x and y directions
        """
        self.ax, self.bx = ax, bx
        self.ay, self.by = ay, by
        self.nx, self.ny = nx, ny

        self.hx = (bx - ax) / (nx + 1)
        self.hy = (by - ay) / (ny + 1)

        self.x = np.linspace(ax, bx, nx + 2)
        self.y = np.linspace(ay, by, ny + 2)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def solve_poisson(self, f, bc_func=None):
        """
        Solve 2D Poisson equation: -Δu = f(x, y)

        Parameters:
        -----------
        f : callable
            Right-hand side function f(x, y)
        bc_func : callable or float
            Boundary condition (default: 0)

        Returns:
        --------
        U : ndarray (ny+2, nx+2)
            Solution array (including boundary)
        """
        if bc_func is None:
            bc_func = lambda x, y: 0.0

        # Build 2D Laplacian matrix using Kronecker products
        # Δu ≈ (u_{i-1,j} - 2u_{i,j} + u_{i+1,j})/hx^2
        #    + (u_{i,j-1} - 2u_{i,j} + u_{i,j+1})/hy^2

        # 1D second derivative operators
        Dx = diags([-1, 2, -1], [-1, 0, 1], shape=(self.nx, self.nx)) / (self.hx**2)
        Dy = diags([-1, 2, -1], [-1, 0, 1], shape=(self.ny, self.ny)) / (self.hy**2)

        # 2D Laplacian: L = Ix ⊗ Dy + Dx ⊗ Iy
        Ix = eye(self.nx)
        Iy = eye(self.ny)
        L = kron(Iy, Dx) + kron(Dy, Ix)

        # Right-hand side (interior points only)
        X_int = self.X[1:-1, 1:-1]
        Y_int = self.Y[1:-1, 1:-1]
        f_int = f(X_int, Y_int).flatten()

        # Apply boundary conditions (add to RHS)
        # TODO: Implement proper BC handling for non-zero BCs

        # Solve linear system
        u_int = spsolve(L.tocsr(), f_int)

        # Reshape and add boundary values
        U = np.zeros((self.ny + 2, self.nx + 2))
        U[1:-1, 1:-1] = u_int.reshape((self.ny, self.nx))

        # Set boundary values
        if callable(bc_func):
            U[0, :] = bc_func(self.x, self.ay)    # Bottom
            U[-1, :] = bc_func(self.x, self.by)   # Top
            U[:, 0] = bc_func(self.ax, self.y)    # Left
            U[:, -1] = bc_func(self.bx, self.y)   # Right
        else:
            U[0, :] = U[-1, :] = U[:, 0] = U[:, -1] = bc_func

        return U

    def plot_solution(self, U, title="FDM Solution", save_path=None):
        """
        Plot 2D solution as contour plot.

        Parameters:
        -----------
        U : ndarray
            Solution array
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        contour = ax.contourf(self.X, self.Y, U, levels=20, cmap='viridis')
        ax.contour(self.X, self.Y, U, levels=10, colors='black', linewidths=0.5, alpha=0.3)

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('u(x, y)')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def test_fdm_1d():
    """
    Test FDM solver on 1D Poisson equation with manufactured solution.

    Problem: -u'' = π² sin(πx) on [0, 1]
    Exact solution: u(x) = sin(πx)
    """
    print("=== Testing 1D FDM Solver ===\n")

    # Define problem
    f = lambda x: np.pi**2 * np.sin(np.pi * x)
    u_exact = lambda x: np.sin(np.pi * x)

    # Solve
    solver = FDMSolver1D(a=0.0, b=1.0, n=100)
    u_h = solver.solve_poisson(f, ua=0.0, ub=0.0)

    # Compute error
    u_ex = u_exact(solver.x)
    error = np.sqrt(solver.h * np.sum((u_h - u_ex)**2))
    print(f"L2 error: {error:.6e}\n")

    # Convergence study
    print("Convergence study:")
    h_vals, errors = solver.convergence_study(f, u_exact)

    for h, err in zip(h_vals, errors):
        print(f"h = {h:.4f}, error = {err:.6e}")

    # Check convergence rate (should be O(h^2))
    rates = np.log(errors[:-1] / errors[1:]) / np.log(h_vals[:-1] / h_vals[1:])
    print(f"\nConvergence rates: {rates}")
    print(f"Expected: ~2.0 (O(h^2))")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(solver.x, u_ex, 'k-', label='Exact', linewidth=2)
    ax1.plot(solver.x, u_h, 'r--', label='FDM (n=100)', linewidth=1.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title('1D Poisson: FDM vs Exact Solution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.loglog(h_vals, errors, 'bo-', label='FDM error', markersize=8)
    ax2.loglog(h_vals, h_vals**2, 'k--', label='O(h²)', linewidth=1.5)
    ax2.set_xlabel('Mesh spacing (h)')
    ax2.set_ylabel('L² Error')
    ax2.set_title('Convergence Study')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    from pathlib import Path
    fig_path = Path(__file__).parent.parent / 'figures' / 'fig2_fdm_1d_validation.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {fig_path}")
    plt.close()


def test_fdm_2d():
    """
    Test FDM solver on 2D Poisson equation with manufactured solution.

    Problem: -Δu = 2π² sin(πx) sin(πy) on [0, 1] × [0, 1]
    Exact solution: u(x, y) = sin(πx) sin(πy)
    """
    print("\n=== Testing 2D FDM Solver ===\n")

    # Define problem
    f = lambda x, y: 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    u_exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)

    # Solve
    solver = FDMSolver2D(ax=0.0, bx=1.0, ay=0.0, by=1.0, nx=50, ny=50)
    U_h = solver.solve_poisson(f, bc_func=0.0)

    # Compute error
    U_ex = u_exact(solver.X, solver.Y)
    error = np.sqrt(solver.hx * solver.hy * np.sum((U_h - U_ex)**2))
    print(f"L2 error: {error:.6e}\n")

    # Plot
    from pathlib import Path
    fig_path = Path(__file__).parent.parent / 'figures' / 'fig3_fdm_2d_solution.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    solver.plot_solution(U_h, title="2D Poisson: FDM Solution",
                        save_path=str(fig_path))
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    # Run tests
    test_fdm_1d()
    test_fdm_2d()

    print("\n✓ FDM solver tests completed!")
