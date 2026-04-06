"""
Sensitivity Analysis for Material Property Variations
Analyzes how stress varies with changes in lamina cribrosa stiffness
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def sensitivity_study_elasticity():
    """
    Vary E_LC (lamina cribrosa Young's modulus) by ±50%
    and observe effect on maximum stress
    """

    # Baseline parameters
    E_LC_baseline = 0.2e6  # Pa (200 kPa)
    IOP_Pa = 30 * 133.322  # 30 mmHg IOP (mild glaucoma scenario)

    # Stress concentration factor (calibrated)
    stress_concentration = 6.2

    # Range of E_LC values (±50%)
    E_LC_variations = np.linspace(0.5, 1.5, 21) * E_LC_baseline  # 50% below to 50% above

    results = []

    for E_LC in E_LC_variations:
        # Simplified stress model
        # σ = P × stress_concentration
        # Strain: ε = σ / E
        # Displacement: u ∝ σ / E

        sigma_max_Pa = IOP_Pa * stress_concentration
        sigma_max_kPa = sigma_max_Pa * 1e-3

        # Strain (dimensionless)
        strain = sigma_max_Pa / E_LC

        # Displacement (rough estimate, μm)
        # u ≈ σ * L / E, where L ~ 0.2 mm (LC thickness)
        L = 0.2  # mm
        displacement_mm = (sigma_max_Pa / E_LC) * L
        displacement_um = displacement_mm * 1000

        results.append({
            'E_LC': E_LC,
            'E_LC_ratio': E_LC / E_LC_baseline,
            'sigma_max_kPa': sigma_max_kPa,
            'strain_percent': strain * 100,
            'displacement_um': displacement_um
        })

    return results, E_LC_baseline


def create_sensitivity_figure():
    """Generate Figure 7: Sensitivity analysis"""

    results, E_baseline = sensitivity_study_elasticity()

    # Extract data
    E_ratios = [r['E_LC_ratio'] for r in results]
    stresses = [r['sigma_max_kPa'] for r in results]
    strains = [r['strain_percent'] for r in results]
    displacements = [r['displacement_um'] for r in results]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Stress vs E_LC (constant - stress doesn't depend on E in this simplified model)
    # But strain and displacement do!
    ax1.plot(E_ratios, strains, 'b-o', linewidth=2, markersize=6, label='Maximum Strain')
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline ($E_{LC}$ = 200 kPa)')
    ax1.fill_between([0.5, 1.5], 0, max(strains)*1.1, alpha=0.1, color='gray')

    ax1.set_xlabel('Relative Stiffness ($E_{LC}$ / $E_{baseline}$)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Maximum Strain (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Sensitivity to Lamina Cribrosa Stiffness\n(IOP = 30 mmHg)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.45, 1.55])

    # Add annotations
    idx_soft = 0
    idx_baseline = len(E_ratios) // 2
    idx_stiff = -1

    ax1.annotate(f'{strains[idx_soft]:.1f}%',
                xy=(E_ratios[idx_soft], strains[idx_soft]),
                xytext=(0.6, strains[idx_soft] + 5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', fc='yellow', alpha=0.7))

    ax1.annotate(f'{strains[idx_baseline]:.1f}%',
                xy=(E_ratios[idx_baseline], strains[idx_baseline]),
                xytext=(1.0, strains[idx_baseline] + 5),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.7))

    ax1.annotate(f'{strains[idx_stiff]:.1f}%',
                xy=(E_ratios[idx_stiff], strains[idx_stiff]),
                xytext=(1.4, strains[idx_stiff] + 2),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))

    # Plot 2: Displacement vs E_LC
    ax2.plot(E_ratios, displacements, 'r-s', linewidth=2, markersize=6, label='ONH Displacement')
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline')
    ax2.fill_between([0.5, 1.5], 0, max(displacements)*1.1, alpha=0.1, color='gray')

    ax2.set_xlabel('Relative Stiffness ($E_{LC}$ / $E_{baseline}$)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Maximum Displacement (μm)', fontsize=13, fontweight='bold')
    ax2.set_title('ONH Deformation Sensitivity\n(IOP = 30 mmHg)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.45, 1.55])

    # Add annotations
    ax2.annotate(f'{displacements[idx_soft]:.1f} μm',
                xy=(E_ratios[idx_soft], displacements[idx_soft]),
                xytext=(0.6, displacements[idx_soft] + 3),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', fc='yellow', alpha=0.7))

    ax2.annotate(f'{displacements[idx_baseline]:.1f} μm',
                xy=(E_ratios[idx_baseline], displacements[idx_baseline]),
                xytext=(1.0, displacements[idx_baseline] + 3),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.7))

    ax2.annotate(f'{displacements[idx_stiff]:.1f} μm',
                xy=(E_ratios[idx_stiff], displacements[idx_stiff]),
                xytext=(1.4, displacements[idx_stiff] + 1),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', fc='lightblue', alpha=0.7))

    plt.tight_layout()

    # Print summary statistics
    print("\n=== Sensitivity Analysis Summary ===")
    print(f"IOP: 30 mmHg (mild glaucoma)")
    print(f"Baseline E_LC: {E_baseline/1e3:.0f} kPa\n")

    for idx, label in [(0, 'Soft (-50%)'), (len(results)//2, 'Baseline'), (-1, 'Stiff (+50%)')]:
        r = results[idx]
        print(f"{label}:")
        print(f"  E_LC = {r['E_LC']/1e3:.0f} kPa")
        print(f"  Strain = {r['strain_percent']:.2f}%")
        print(f"  Displacement = {r['displacement_um']:.2f} μm")
        print()

    # Calculate sensitivity metric
    strain_change = (strains[0] - strains[-1]) / strains[len(results)//2] * 100
    print(f"Strain sensitivity: {strain_change:.1f}% change over ±50% E_LC variation")

    return fig, results


if __name__ == '__main__':
    # Generate figure
    fig, results = create_sensitivity_figure()

    # Save
    output_dir = Path('/Users/arnavshenoy/Desktop/programming/nhsjs research/pde-paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / 'fig7_sensitivity_analysis.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved figure to {fig_path}")

    # Save numerical results
    results_dir = Path('/Users/arnavshenoy/Desktop/programming/nhsjs research/pde-paper/results')
    with open(results_dir / 'sensitivity_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved numerical data to {results_dir / 'sensitivity_analysis.json'}")
