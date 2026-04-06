"""
2D Eye Geometry Generation for PDE Simulations
Creates simplified axisymmetric models of the eye for biomechanics analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import json
from pathlib import Path


@dataclass
class EyeGeometryParameters:
    """Parameters for 2D axisymmetric eye model"""
    # Global dimensions (mm)
    globe_radius: float = 12.0  # Typical eye radius
    anterior_chamber_depth: float = 3.0
    corneal_radius: float = 7.8

    # Optic nerve head
    onh_diameter: float = 1.7  # From PAPILA dataset
    lamina_cribrosa_thickness: float = 0.2
    scleral_canal_diameter: float = 1.9

    # Tissue thicknesses (mm)
    sclera_thickness: float = 0.5
    cornea_thickness: float = 0.52

    # Material properties
    sclera_E: float = 2.9e6  # Pa (2.9 MPa)
    sclera_nu: float = 0.47
    lamina_E: float = 0.2e6  # Pa (0.2 MPa, mid-range)
    lamina_nu: float = 0.45
    cornea_E: float = 0.3e6  # Pa
    cornea_nu: float = 0.49

    # Fluid properties
    aqueous_viscosity: float = 7e-4  # Pa·s
    aqueous_density: float = 1000.0  # kg/m³
    production_rate: float = 2.5e-9 / 60  # m³/s (2.5 μL/min)

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'globe_radius': self.globe_radius,
            'anterior_chamber_depth': self.anterior_chamber_depth,
            'corneal_radius': self.corneal_radius,
            'onh_diameter': self.onh_diameter,
            'lamina_cribrosa_thickness': self.lamina_cribrosa_thickness,
            'scleral_canal_diameter': self.scleral_canal_diameter,
            'sclera_thickness': self.sclera_thickness,
            'cornea_thickness': self.cornea_thickness,
            'sclera_E': self.sclera_E,
            'sclera_nu': self.sclera_nu,
            'lamina_E': self.lamina_E,
            'lamina_nu': self.lamina_nu,
            'cornea_E': self.cornea_E,
            'cornea_nu': self.cornea_nu,
            'aqueous_viscosity': self.aqueous_viscosity,
            'aqueous_density': self.aqueous_density,
            'production_rate': self.production_rate
        }


class AxisymmetricEyeModel:
    """
    2D axisymmetric representation of the eye
    Coordinate system: (r, z) where r is radial distance from optical axis
    """

    def __init__(self, params: EyeGeometryParameters = None):
        if params is None:
            params = EyeGeometryParameters()
        self.params = params

    def create_rectangular_grid(self, nr: int = 200, nz: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rectangular computational grid

        Args:
            nr: Number of grid points in radial direction
            nz: Number of grid points in axial direction

        Returns:
            r, z: 2D coordinate arrays
        """
        # Domain bounds
        r_max = self.params.globe_radius + 2  # mm, slight buffer
        z_min = -self.params.globe_radius - 2
        z_max = self.params.anterior_chamber_depth + 2

        r = np.linspace(0, r_max, nr)
        z = np.linspace(z_min, z_max, nz)

        R, Z = np.meshgrid(r, z, indexing='ij')

        return R, Z

    def get_tissue_mask(self, R: np.ndarray, Z: np.ndarray) -> dict:
        """
        Create tissue region masks for the geometry

        Returns:
            Dictionary with boolean masks for each tissue type
        """
        masks = {}

        # Sclera: posterior hemisphere
        globe_r = self.params.globe_radius
        sclera_inner = globe_r - self.params.sclera_thickness
        sclera_outer = globe_r

        # Distance from center of posterior globe
        dist_from_center = np.sqrt(R**2 + Z**2)

        masks['sclera'] = (dist_from_center >= sclera_inner) & (dist_from_center <= sclera_outer) & (Z < 0)

        # Lamina cribrosa: disk at optic nerve head
        onh_r = self.params.onh_diameter / 2
        lc_z_start = -globe_r
        lc_z_end = lc_z_start + self.params.lamina_cribrosa_thickness

        masks['lamina_cribrosa'] = (R <= onh_r) & (Z >= lc_z_start) & (Z <= lc_z_end)

        # Cornea: anterior surface (simplified as flat for 2D)
        masks['cornea'] = (R <= self.params.corneal_radius) & (Z >= self.params.anterior_chamber_depth - 0.1) & (Z <= self.params.anterior_chamber_depth + self.params.cornea_thickness)

        # Aqueous humor: anterior chamber
        masks['aqueous'] = (R <= self.params.corneal_radius) & (Z >= 0) & (Z <= self.params.anterior_chamber_depth)

        # Interior (vitreous/retina - simplified)
        masks['interior'] = (dist_from_center < sclera_inner) & (Z < 0) & ~masks['lamina_cribrosa']

        return masks

    def get_material_properties(self, R: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spatially-varying material properties

        Returns:
            E_field: Young's modulus at each point
            nu_field: Poisson's ratio at each point
        """
        masks = self.get_tissue_mask(R, Z)

        E_field = np.zeros_like(R)
        nu_field = np.zeros_like(R)

        # Assign properties based on tissue type
        E_field[masks['sclera']] = self.params.sclera_E
        E_field[masks['lamina_cribrosa']] = self.params.lamina_E
        E_field[masks['cornea']] = self.params.cornea_E

        nu_field[masks['sclera']] = self.params.sclera_nu
        nu_field[masks['lamina_cribrosa']] = self.params.lamina_nu
        nu_field[masks['cornea']] = self.params.cornea_nu

        return E_field, nu_field

    def visualize_geometry(self, R: np.ndarray, Z: np.ndarray, save_path=None):
        """Visualize the 2D geometry and tissue regions"""
        masks = self.get_tissue_mask(R, Z)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create composite image showing different tissues
        composite = np.zeros_like(R)
        composite[masks['sclera']] = 1
        composite[masks['lamina_cribrosa']] = 2
        composite[masks['cornea']] = 3
        composite[masks['aqueous']] = 4
        composite[masks['interior']] = 5

        im = ax.contourf(R, Z, composite, levels=[0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                        colors=['white', 'brown', 'gray', 'lightblue', 'lightcyan', 'lightyellow'],
                        alpha=0.7)

        # Add contour lines
        ax.contour(R, Z, composite, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='black', linewidths=1)

        # Labels and annotations
        ax.set_xlabel('Radial Distance r (mm)', fontsize=12)
        ax.set_ylabel('Axial Distance z (mm)', fontsize=12)
        ax.set_title('2D Axisymmetric Eye Geometry', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='brown', alpha=0.7, label='Sclera'),
            Patch(facecolor='gray', alpha=0.7, label='Lamina Cribrosa'),
            Patch(facecolor='lightblue', alpha=0.7, label='Cornea'),
            Patch(facecolor='lightcyan', alpha=0.7, label='Aqueous Humor'),
            Patch(facecolor='lightyellow', alpha=0.7, label='Interior (Vitreous)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Geometry visualization saved to {save_path}")

        return fig, ax


def create_onh_focused_geometry(params: EyeGeometryParameters,
                                 nr: int = 150, nz: int = 150) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a zoomed-in geometry focused on the optic nerve head region

    Returns:
        R, Z: Coordinate grids
        info: Dictionary with geometry information
    """
    # Focused domain around ONH
    r_max = params.onh_diameter * 2  # 2x ONH diameter
    z_center = -params.globe_radius
    z_range = params.onh_diameter * 1.5

    r = np.linspace(0, r_max, nr)
    z = np.linspace(z_center - z_range/2, z_center + z_range/2, nz)

    R, Z = np.meshgrid(r, z, indexing='ij')

    # Create tissue masks for ONH region
    masks = {}

    # Lamina cribrosa
    lc_r = params.onh_diameter / 2
    lc_z_start = z_center
    lc_z_end = lc_z_start + params.lamina_cribrosa_thickness
    masks['lamina_cribrosa'] = (R <= lc_r) & (Z >= lc_z_start) & (Z <= lc_z_end)

    # Sclera around canal
    canal_r = params.scleral_canal_diameter / 2
    sclera_thickness = params.sclera_thickness
    masks['sclera'] = (R >= canal_r) & (R <= canal_r + sclera_thickness)

    # Neural tissue (simplified)
    masks['neural_tissue'] = (R <= lc_r) & (Z < lc_z_start)

    info = {
        'r_max': r_max,
        'z_center': z_center,
        'z_range': z_range,
        'masks': masks
    }

    return R, Z, info


if __name__ == '__main__':
    # Create default geometry
    params = EyeGeometryParameters()

    # Save parameters
    output_dir = Path('/Users/arnavshenoy/Desktop/programming/nhsjs research/pde-paper/data')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'geometry_parameters.json', 'w') as f:
        json.dump(params.to_dict(), f, indent=2)

    print("Geometry parameters saved to:", output_dir / 'geometry_parameters.json')

    # Create and visualize full geometry
    model = AxisymmetricEyeModel(params)
    R, Z = model.create_rectangular_grid(nr=200, nz=200)

    fig_dir = Path('/Users/arnavshenoy/Desktop/programming/nhsjs research/pde-paper/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)

    model.visualize_geometry(R, Z, save_path=fig_dir / 'fig1_geometry.png')

    # Create ONH-focused geometry
    R_onh, Z_onh, info_onh = create_onh_focused_geometry(params)
    print(f"\nONH-focused grid created: {R_onh.shape}")
    print(f"Domain: r ∈ [0, {info_onh['r_max']:.2f}] mm")
    print(f"        z ∈ [{info_onh['z_center'] - info_onh['z_range']/2:.2f}, "
          f"{info_onh['z_center'] + info_onh['z_range']/2:.2f}] mm")

    print("\nGeometry generation complete!")
