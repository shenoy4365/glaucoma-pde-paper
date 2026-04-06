"""
Aqueous Humor Pressure Distribution Simulation
Solves simplified Poisson equation for steady-state pressure in anterior chamber
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from fdm_solver import FDMSolver2D


def solve_anterior_chamber_pressure():
    """
    Solve for aqueous humor pressure distribution in simplified anterior chamber

    Domain: Rectangular approximation of anterior chamber
    - Width: 12 mm (approximate corneal diameter)
    - Height: 3 mm (anterior chamber depth)

    PDE: ∇²p = f
    where f represents production (ciliary body) and drainage (trabecular meshwork)
    """

    # Domain parameters (mm)
    width = 12.0
    height = 3.0

    # Discretization
    nx, ny = 120, 30

    # Create solver
    solver = FDMSolver2D(ax=0, bx=width, ay=0, by=height, nx=nx, ny=ny)

    # Source/sink term
    # Production at ciliary body (posterior, y ~ 0)
    # Drainage at trabecular meshwork (peripheral, x ~ width)

    def source_term(X, Y):
        """
        Source term for aqueous humor production/drainage

        Positive: Production (ciliary body at periphery)
        Negative: Drainage (trabecular meshwork)
        """
        f = np.zeros_like(X)

        # Production region: lower boundary, peripheral
        production_mask = (Y < 0.5) & (X > width * 0.7)
        f[production_mask] = 100.0  # Arbitrary units (Pa/mm²)

        # Drainage region: upper peripheral (trabecular meshwork)
        drainage_mask = (Y > height * 0.8) & (X > width * 0.75)
        f[drainage_mask] = -150.0

        return f

    # Boundary conditions: atmospheric pressure at trabecular meshwork
    # Simplified: p = 0 at boundaries (gauge pressure)
    def bc_func(x, y):
        return 0.0

    # Solve
    P = solver.solve_poisson(source_term, bc_func=bc_func)

    # Add baseline IOP (15 mmHg = 2000 Pa)
    baseline_IOP = 2000  # Pa
    P_total = P + baseline_IOP

    return solver, P_total


def create_pressure_figure():
    """Generate Figure 4: Pressure field in anterior chamber"""

    solver, P = solve_anterior_chamber_pressure()

    # Convert to mmHg
    P_mmHg = P / 133.322

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Pressure contours
    levels = np.linspace(P_mmHg.min(), P_mmHg.max(), 20)
    contour = ax1.contourf(solver.X, solver.Y, P_mmHg, levels=levels, cmap='coolwarm')
    ax1.contour(solver.X, solver.Y, P_mmHg, levels=10, colors='black', linewidths=0.5, alpha=0.3)

    cbar1 = plt.colorbar(contour, ax=ax1)
    cbar1.set_label('Pressure (mmHg)', fontsize=12)

    ax1.set_xlabel('Horizontal Distance (mm)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Anterior Chamber Depth (mm)', fontsize=13, fontweight='bold')
    ax1.set_title('Aqueous Humor Pressure Distribution\n(Steady-State, IOP = 15 mmHg)',
                  fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # Add annotations
    ax1.text(9, 0.3, 'Production\n(Ciliary Body)', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax1.text(9, 2.5, 'Drainage\n(Trabecular\nMeshwork)', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Plot 2: Pressure gradient magnitude
    # Compute gradient
    dpdx = np.gradient(P, solver.hx, axis=1)
    dpdy = np.gradient(P, solver.hy, axis=0)
    grad_mag = np.sqrt(dpdx**2 + dpdy**2)  # Pa/mm

    contour2 = ax2.contourf(solver.X, solver.Y, grad_mag, levels=20, cmap='hot')
    ax2.contour(solver.X, solver.Y, grad_mag, levels=10, colors='black', linewidths=0.5, alpha=0.3)

    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('Pressure Gradient Magnitude (Pa/mm)', fontsize=12)

    ax2.set_xlabel('Horizontal Distance (mm)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Anterior Chamber Depth (mm)', fontsize=13, fontweight='bold')
    ax2.set_title('Pressure Gradient\n(Drives Aqueous Flow)',
                  fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    # Compute statistics
    delta_p = P_mmHg.max() - P_mmHg.min()
    max_grad = grad_mag.max()

    print("\n=== Pressure Field Statistics ===")
    print(f"Average pressure: {P_mmHg.mean():.2f} mmHg")
    print(f"Pressure variation: {delta_p:.3f} mmHg ({delta_p/P_mmHg.mean()*100:.2f}%)")
    print(f"Maximum pressure gradient: {max_grad:.1f} Pa/mm")

    return fig, P_mmHg, grad_mag


if __name__ == '__main__':
    # Generate figure
    fig, P, grad = create_pressure_figure()

    # Save
    output_dir = Path('/Users/arnavshenoy/Desktop/programming/nhsjs research/pde-paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / 'fig4_pressure_distribution.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved figure to {fig_path}")
