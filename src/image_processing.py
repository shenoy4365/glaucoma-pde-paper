"""
Image Processing for Optic Disc Parameter Extraction
Extracts geometric parameters from fundus images for PDE model validation
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import json


class OpticDiscSegmenter:
    """Extract optic disc and cup parameters from fundus images"""

    def __init__(self, pixel_to_mm: float = 0.01):
        """
        Args:
            pixel_to_mm: Conversion factor from pixels to mm (approximate)
                        Typical fundus image: ~1500 pixels ≈ 15 mm field of view
        """
        self.pixel_to_mm = pixel_to_mm

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess fundus image for disc segmentation"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        return blurred

    def detect_optic_disc(self, image: np.ndarray) -> Tuple[int, int, float]:
        """
        Detect optic disc using bright region detection

        Returns:
            (center_x, center_y, radius_pixels)
        """
        processed = self.preprocess_image(image)

        # Threshold to find bright regions (optic disc is brightest)
        _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, None

        # Find largest bright region (likely optic disc)
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        return int(x), int(y), radius

    def detect_optic_cup(self, image: np.ndarray, disc_center: Tuple[int, int],
                         disc_radius: float) -> float:
        """
        Detect optic cup (central excavation) within disc

        Returns:
            cup_radius_pixels
        """
        processed = self.preprocess_image(image)

        # Extract region of interest around disc
        x, y = disc_center
        r = int(disc_radius * 1.2)  # Slightly larger region

        roi = processed[max(0, y-r):y+r, max(0, x-r):x+r]

        if roi.size == 0:
            return disc_radius * 0.5  # Default CDR ≈ 0.5

        # Cup is typically darker region in center
        _, cup_thresh = cv2.threshold(roi, np.mean(roi) * 0.7, 255, cv2.THRESH_BINARY_INV)

        cup_contours, _ = cv2.findContours(cup_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cup_contours:
            largest_cup = max(cup_contours, key=cv2.contourArea)
            _, cup_radius = cv2.minEnclosingCircle(largest_cup)
            return cup_radius
        else:
            return disc_radius * 0.5  # Default

    def extract_parameters(self, image_path: Path) -> Dict[str, float]:
        """
        Extract all relevant parameters from a single image

        Returns:
            {
                'disc_diameter_mm': float,
                'cup_diameter_mm': float,
                'cdr': float,  # Cup-to-disc ratio
                'disc_area_mm2': float,
                'quality': float  # Segmentation quality (0-1)
            }
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        # Detect optic disc
        disc_x, disc_y, disc_radius = self.detect_optic_disc(image)

        if disc_radius is None:
            return None

        # Detect optic cup
        cup_radius = self.detect_optic_cup(image, (disc_x, disc_y), disc_radius)

        # Convert to mm
        disc_diameter_mm = 2 * disc_radius * self.pixel_to_mm
        cup_diameter_mm = 2 * cup_radius * self.pixel_to_mm

        # Calculate CDR
        cdr = cup_radius / disc_radius if disc_radius > 0 else 0.5

        # Calculate area
        disc_area_mm2 = np.pi * (disc_diameter_mm / 2) ** 2

        # Quality heuristic: penalize if disc too small/large
        quality = 1.0
        if disc_diameter_mm < 1.2 or disc_diameter_mm > 2.5:  # Normal range 1.5-2.0 mm
            quality = 0.5

        return {
            'disc_diameter_mm': disc_diameter_mm,
            'cup_diameter_mm': cup_diameter_mm,
            'cdr': cdr,
            'disc_area_mm2': disc_area_mm2,
            'quality': quality
        }


def process_hygd_dataset(dataset_path: Path, max_images: int = 200,
                         output_path: Path = None) -> pd.DataFrame:
    """
    Process HYGD dataset and extract geometric parameters

    Args:
        dataset_path: Path to 'hf dataset' folder
        max_images: Maximum number of images to process (for efficiency)
        output_path: Where to save results CSV

    Returns:
        DataFrame with extracted parameters
    """
    print(f"Processing HYGD dataset from {dataset_path}")

    # Load labels
    labels_df = pd.read_csv(dataset_path / 'Labels.csv')
    print(f"Found {len(labels_df)} images in Labels.csv")

    # Filter high-quality images (quality score > 4.0)
    quality_threshold = 4.0
    high_quality = labels_df[labels_df['Quality Score'] > quality_threshold]
    print(f"High-quality images (score > {quality_threshold}): {len(high_quality)}")

    # Sample subset
    if len(high_quality) > max_images:
        sample_df = high_quality.sample(n=max_images, random_state=42)
    else:
        sample_df = high_quality

    print(f"Processing {len(sample_df)} images...")

    # Initialize segmenter
    segmenter = OpticDiscSegmenter(pixel_to_mm=0.01)

    results = []
    images_path = dataset_path / 'Images'

    for idx, row in sample_df.iterrows():
        image_name = row['Image Name']
        image_path = images_path / image_name

        if not image_path.exists():
            continue

        # Extract parameters
        params = segmenter.extract_parameters(image_path)

        if params is not None:
            params['image_name'] = image_name
            params['patient'] = row['Patient']
            params['label'] = row['Label']
            params['image_quality'] = row['Quality Score']
            results.append(params)

        if len(results) % 50 == 0:
            print(f"Processed {len(results)} images...")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return results_df


def compute_statistics(results_df: pd.DataFrame) -> Dict:
    """Compute summary statistics for extracted parameters"""

    # Separate by glaucoma status
    gon_positive = results_df[results_df['label'] == 'GON+']
    gon_negative = results_df[results_df['label'] == 'GON-']

    stats = {
        'overall': {
            'n_images': len(results_df),
            'disc_diameter_mean': results_df['disc_diameter_mm'].mean(),
            'disc_diameter_std': results_df['disc_diameter_mm'].std(),
            'cdr_mean': results_df['cdr'].mean(),
            'cdr_std': results_df['cdr'].std(),
            'disc_area_mean': results_df['disc_area_mm2'].mean(),
        },
        'glaucoma': {
            'n_images': len(gon_positive),
            'disc_diameter_mean': gon_positive['disc_diameter_mm'].mean(),
            'cdr_mean': gon_positive['cdr'].mean(),
            'cdr_std': gon_positive['cdr'].std(),
        },
        'normal': {
            'n_images': len(gon_negative),
            'disc_diameter_mean': gon_negative['disc_diameter_mm'].mean(),
            'cdr_mean': gon_negative['cdr'].mean(),
            'cdr_std': gon_negative['cdr'].std(),
        } if len(gon_negative) > 0 else None
    }

    return stats


def visualize_sample(dataset_path: Path, results_df: pd.DataFrame, n_samples: int = 6):
    """Visualize sample segmentations"""

    sample = results_df.sample(n=min(n_samples, len(results_df)), random_state=42)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    images_path = dataset_path / 'Images'

    for idx, (_, row) in enumerate(sample.iterrows()):
        if idx >= n_samples:
            break

        image = cv2.imread(str(images_path / row['image_name']))
        if image is None:
            continue

        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(image_rgb)
        axes[idx].set_title(
            f"{row['label']}\n"
            f"CDR: {row['cdr']:.2f}, Disc: {row['disc_diameter_mm']:.2f}mm"
        )
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Configuration
    dataset_path = Path('/Users/arnavshenoy/Desktop/programming/nhsjs research/hf dataset')
    output_dir = Path('/Users/arnavshenoy/Desktop/programming/nhsjs research/pde-paper/data')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process dataset
    results_df = process_hygd_dataset(
        dataset_path=dataset_path,
        max_images=200,
        output_path=output_dir / 'extracted_parameters.csv'
    )

    # Compute statistics
    stats = compute_statistics(results_df)

    # Save statistics
    with open(output_dir / 'parameter_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total images processed: {stats['overall']['n_images']}")
    print(f"\nAverage optic disc diameter: {stats['overall']['disc_diameter_mean']:.2f} ± {stats['overall']['disc_diameter_std']:.2f} mm")
    print(f"Average CDR: {stats['overall']['cdr_mean']:.2f} ± {stats['overall']['cdr_std']:.2f}")
    print(f"Average disc area: {stats['overall']['disc_area_mean']:.2f} mm²")

    if stats['glaucoma']['n_images'] > 0:
        print(f"\n--- Glaucomatous Eyes (GON+) ---")
        print(f"N = {stats['glaucoma']['n_images']}")
        print(f"CDR: {stats['glaucoma']['cdr_mean']:.2f} ± {stats['glaucoma']['cdr_std']:.2f}")

    if stats['normal'] is not None:
        print(f"\n--- Normal Eyes (GON-) ---")
        print(f"N = {stats['normal']['n_images']}")
        print(f"CDR: {stats['normal']['cdr_mean']:.2f} ± {stats['normal']['cdr_std']:.2f}")

    print("\nParameters saved to:", output_dir / 'extracted_parameters.csv')
    print("Statistics saved to:", output_dir / 'parameter_statistics.json')
