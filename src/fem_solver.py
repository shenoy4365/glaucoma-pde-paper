"""
Finite Element Method Solver using FEniCS
Implements Poisson and linear elasticity solvers for glaucoma modeling
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from fenics import *
    FENICS_AVAILABLE = True
except ImportError:
    print("Warning: FEniCS not available. Install with: conda install -c conda-forge fenics")
    FENICS_AVAILABLE = False


class FEMPoissonSolver:
    """Finite Element solver for Poisson equation using FEniCS"""

    def __init__(self, mesh):
        """
        Args:
            mesh: FEniCS mesh object
        """
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS is required but not installed")

        self.mesh = mesh
        self.V = FunctionSpace(mesh, 'P', 2)  # Quadratic Lagrange elements

    def solve(self, f_expr, boundary_conditions, solver_type='direct'):
        """
        Solve -∆u = f with specified boundary conditions

        Args:
            f_expr: Source term as FEniCS Expression or Constant
            boundary_conditions: List of (subdomain, value) tuples
            solver_type: 'direct' or 'iterative'

        Returns:
            FEniCS Function object containing solution
        """
        # Define variational problem
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        # Weak form: ∫ ∇u·∇v dx = ∫ f v dx
        a = dot(grad(u), grad(v)) * dx
        L = f_expr * v * dx

        # Apply boundary conditions
        bcs = []
        for subdomain, value in boundary_conditions:
            bc = DirichletBC(self.V, value, subdomain)
            bcs.append(bc)

        # Solve
        u_h = Function(self.V)
        if solver_type == 'direct':
            solve(a == L, u_h, bcs)
        else:
            # Iterative solver with preconditioning
            solve(a == L, u_h, bcs,
                  solver_parameters={'linear_solver': 'gmres',
                                   'preconditioner': 'amg'})

        return u_h

    def compute_error(self, u_h, u_exact):
        """
        Compute L2 error between numerical and exact solution

        Args:
            u_h: Numerical solution (FEniCS Function)
            u_exact: Exact solution (FEniCS Expression)

        Returns:
            L2 error norm
        """
        error_L2 = errornorm(u_exact, u_h, 'L2')
        return error_L2


class FEMElasticitySolver:
    """Finite Element solver for linear elasticity"""

    def __init__(self, mesh, E, nu):
        """
        Args:
            mesh: FEniCS mesh
            E: Young's modulus (can be spatially varying)
            nu: Poisson's ratio
        """
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS is required")

        self.mesh = mesh
        self.E = E
        self.nu = nu

        # Lamé parameters
        self.mu = Constant(E / (2 * (1 + nu)))
        self.lmbda = Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))

        # Vector function space for displacement
        self.V = VectorFunctionSpace(mesh, 'P', 2)

    def epsilon(self, u):
        """Strain tensor"""
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

    def sigma(self, u):
        """Stress tensor (linear elasticity)"""
        return self.lmbda * tr(self.epsilon(u)) * Identity(u.geometric_dimension()) + 2 * self.mu * self.epsilon(u)

    def solve(self, body_force, boundary_conditions, traction_conditions=None):
        """
        Solve linear elasticity: -div(σ) = f

        Args:
            body_force: Body force vector (e.g., gravity)
            boundary_conditions: List of (subdomain, displacement_value) tuples
            traction_conditions: List of (boundary_part, traction_vector) tuples

        Returns:
            Displacement field u, stress function
        """
        # Define variational problem
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        # Weak form: ∫ σ(u):ε(v) dx = ∫ f·v dx + ∫ T·v ds
        a = inner(self.sigma(u), self.epsilon(v)) * dx
        L = dot(body_force, v) * dx

        # Add traction boundary conditions
        if traction_conditions:
            for boundary_part, traction in traction_conditions:
                L += dot(traction, v) * boundary_part

        # Apply Dirichlet boundary conditions
        bcs = []
        for subdomain, value in boundary_conditions:
            bc = DirichletBC(self.V, value, subdomain)
            bcs.append(bc)

        # Solve
        u_h = Function(self.V)
        solve(a == L, u_h, bcs)

        # Compute stress
        W = TensorFunctionSpace(self.mesh, 'P', 1)
        sigma_h = project(self.sigma(u_h), W)

        return u_h, sigma_h

    def von_mises_stress(self, sigma_h):
        """
        Compute von Mises stress from stress tensor

        σ_vM = sqrt(3/2 σ':σ') where σ' is deviatoric stress
        """
        s = sigma_h - (1. / 3) * tr(sigma_h) * Identity(sigma_h.geometric_dimension())
        von_mises = sqrt(3. / 2 * inner(s, s))

        V_scalar = FunctionSpace(self.mesh, 'P', 1)
        von_mises_proj = project(von_mises, V_scalar)

        return von_mises_proj


def create_1d_mesh(n_elements):
    """Create 1D interval mesh"""
    if not FENICS_AVAILABLE:
        return None
    return UnitIntervalMesh(n_elements)


def create_2d_rectangular_mesh(nx, ny):
    """Create 2D rectangular mesh"""
    if not FENICS_AVAILABLE:
        return None
    return UnitSquareMesh(nx, ny)


def create_2d_disk_mesh(center, radius, resolution=50):
    """Create 2D disk mesh using mshr (if available)"""
    if not FENICS_AVAILABLE:
        return None

    try:
        from mshr import Circle, generate_mesh
        domain = Circle(Point(center[0], center[1]), radius)
        mesh = generate_mesh(domain, resolution)
        return mesh
    except ImportError:
        print("Warning: mshr not available for custom geometries")
        return None


# ============================================================
# Validation Tests
# ============================================================

def test_1d_poisson():
    """
    Test 1D Poisson solver with manufactured solution
    Exact: u(x) = sin(πx), f(x) = π²sin(πx)
    """
    if not FENICS_AVAILABLE:
        print("Skipping test: FEniCS not available")
        return

    print("\n=== 1D Poisson Validation (FEM) ===")

    errors = []
    h_values = []

    for n in [10, 20, 40, 80]:
        mesh = UnitIntervalMesh(n)
        solver = FEMPoissonSolver(mesh)

        # Manufactured solution
        u_exact = Expression('sin(pi*x[0])', degree=5)
        f = Expression('pi*pi*sin(pi*x[0])', degree=5)

        # Boundary conditions
        def boundary(x, on_boundary):
            return on_boundary

        bcs = [(boundary, Constant(0.0))]

        # Solve
        u_h = solver.solve(f, bcs)

        # Compute error
        error_L2 = solver.compute_error(u_h, u_exact)

        h = 1.0 / n
        errors.append(error_L2)
        h_values.append(h)

        print(f"n={n:3d}, h={h:.4f}, L2 error = {error_L2:.3e}")

    # Compute convergence rates
    print("\nConvergence rates:")
    for i in range(1, len(errors)):
        rate = np.log(errors[i] / errors[i-1]) / np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.4f}: rate = {rate:.2f}")

    return h_values, errors


def test_2d_poisson():
    """
    Test 2D Poisson solver
    Exact: u(x,y) = sin(πx)sin(πy)
    """
    if not FENICS_AVAILABLE:
        print("Skipping test: FEniCS not available")
        return

    print("\n=== 2D Poisson Validation (FEM) ===")

    errors = []
    h_values = []
    dofs = []

    for n in [10, 20, 40, 80]:
        mesh = UnitSquareMesh(n, n)
        solver = FEMPoissonSolver(mesh)

        # Manufactured solution
        u_exact = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=5)
        f = Expression('2*pi*pi*sin(pi*x[0])*sin(pi*x[1])', degree=5)

        # Boundary conditions
        def boundary(x, on_boundary):
            return on_boundary

        bcs = [(boundary, Constant(0.0))]

        # Solve
        u_h = solver.solve(f, bcs)

        # Compute error
        error_L2 = solver.compute_error(u_h, u_exact)

        h = 1.0 / n
        errors.append(error_L2)
        h_values.append(h)
        dofs.append(solver.V.dim())

        print(f"n={n:3d}, DOF={dofs[-1]:5d}, h={h:.5f}, L2 error = {error_L2:.3e}")

    # Convergence rates
    print("\nConvergence rates:")
    for i in range(1, len(errors)):
        rate = np.log(errors[i] / errors[i-1]) / np.log(h_values[i] / h_values[i-1])
        print(f"  h={h_values[i]:.5f}: rate = {rate:.2f}")

    return h_values, errors, dofs


if __name__ == '__main__':
    if FENICS_AVAILABLE:
        # Run validation tests
        h1, e1 = test_1d_poisson()
        h2, e2, dof2 = test_2d_poisson()

        # Save results
        output_dir = Path('/Users/arnavshenoy/Desktop/programming/nhsjs research/pde-paper/results')
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez(output_dir / 'fem_validation_1d.npz',
                 h=h1, errors=e1)
        np.savez(output_dir / 'fem_validation_2d.npz',
                 h=h2, errors=e2, dofs=dof2)

        print("\nResults saved to:", output_dir)
    else:
        print("FEniCS not available. Please install: conda install -c conda-forge fenics")
