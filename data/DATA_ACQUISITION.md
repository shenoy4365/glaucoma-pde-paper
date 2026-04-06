# Dataset Acquisition Guide

This document provides step-by-step instructions for downloading the three publicly available glaucoma datasets used in this research.

## Overview

This project uses three open-access datasets containing fundus images, clinical measurements, and longitudinal IOP data:

| Dataset | Images/Patients | License | Key Parameters |
|---------|----------------|---------|----------------|
| HYGD    | 747 fundus images | CC BY 4.0 | Expert-annotated glaucoma labels |
| PAPILA  | 488 bilateral images (244 patients) | CC BY 4.0 | CDR, optic disc/cup segmentations |
| GRAPE   | 264 patients | CC BY 4.0 | Longitudinal IOP, visual fields |

All datasets are freely available under Creative Commons Attribution 4.0 International licenses.

---

## 1. HYGD - Hillel Yaffe Glaucoma Dataset

### Description
Gold-standard annotated fundus dataset for glaucoma detection from Hillel Yaffe Medical Center. Contains 747 high-quality fundus photographs with expert annotations.

### Citation
```bibtex
@article{hygd2024,
  author = {Abramovich, Or and Pizem, Hadas and Fhima, Jonathan and Berkowitz, Eran and Gofrit, Ben and Van Eijgen, Jan and Blumenthal, Eytan and Behar, Joachim A.},
  title = {Hillel Yaffe Glaucoma Dataset (HYGD): A gold-standard annotated fundus dataset for glaucoma detection},
  journal = {PhysioNet},
  year = {2025},
  doi = {10.13026/z0ak-km33}
}
```

### Download Instructions

1. **Visit PhysioNet**:
   ```
   https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/
   ```

2. **Create PhysioNet account** (if you don't have one):
   - Go to https://physionet.org/register/
   - Complete registration and sign data use agreement

3. **Download dataset**:
   ```bash
   # Option A: Web download
   # Click "Download the ZIP file" button on the dataset page

   # Option B: Command line (requires PhysioNet credentials)
   wget -r -N -c -np https://physionet.org/files/hillel-yaffe-glaucoma-dataset/1.0.0/
   ```

4. **Extract and organize**:
   ```bash
   cd pde-paper/data/raw/
   mkdir -p hygd
   unzip hillel-yaffe-glaucoma-dataset-1.0.0.zip -d hygd/
   ```

### Expected Structure
```
data/raw/hygd/
├── images/              # 747 fundus images
├── labels.csv           # Glaucoma annotations
└── README.md            # Dataset documentation
```

---

## 2. PAPILA Dataset

### Description
Bilateral fundus images and clinical data of both eyes for glaucoma assessment. Includes expert segmentations of optic disc and cup, enabling cup-to-disc ratio (CDR) extraction.

### Citation
```bibtex
@article{papila2022,
  author = {Kovalyk, Oleksandr and Morales-Sánchez, Juan and Verdú-Monedero, Rafael and Sellés-Navarro, Inmaculada and Palazón-Cabanes, Ana and Sancho-Gómez, José-Luis},
  title = {PAPILA: Dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment},
  journal = {Scientific Data},
  volume = {9},
  pages = {291},
  year = {2022},
  doi = {10.1038/s41597-022-01388-1}
}
```

### Download Instructions

1. **Visit Figshare repository**:
   ```
   https://figshare.com/articles/dataset/PAPILA/14798004
   ```

2. **Download files**:
   - Click "Download all" to get the complete dataset (approximately 3 GB)
   - Alternatively, download individual components:
     - `FundusImages.zip` - Retinal fundus photographs
     - `ExpertsSegmentations.zip` - Optic disc/cup masks
     - `ClinicalData.zip` - Patient metadata and measurements

3. **Extract and organize**:
   ```bash
   cd pde-paper/data/raw/
   mkdir -p papila
   unzip FundusImages.zip -d papila/
   unzip ExpertsSegmentations.zip -d papila/
   unzip ClinicalData.zip -d papila/
   ```

### Expected Structure
```
data/raw/papila/
├── FundusImages/                    # 488 bilateral fundus images
├── ExpertsSegmentations/            # Manual optic disc/cup segmentations
│   ├── OD_Masks/                   # Optic disc masks
│   └── OC_Masks/                   # Optic cup masks
└── ClinicalData/
    └── clinical_data_summary.xlsx   # CDR, IOP, diagnosis labels
```

---

## 3. GRAPE Dataset

### Description
Multi-modal dataset with longitudinal follow-up visual field tests and fundus images for glaucoma management. Contains 264 patients with repeated IOP measurements over time.

### Citation
```bibtex
@article{grape2023,
  author = {Huang, Xiaoling and Kong, Xiangyin and Shen, Ziyan and Ouyang, Jing and Li, Yunxiang and Jin, Kai and Ye, Juan},
  title = {GRAPE: A multi-modal dataset of longitudinal follow-up visual field and fundus images for glaucoma management},
  journal = {Scientific Data},
  volume = {10},
  pages = {520},
  year = {2023},
  doi = {10.1038/s41597-023-02424-4}
}
```

### Download Instructions

1. **Visit Zenodo repository**:
   ```
   https://zenodo.org/record/8003390
   ```

2. **Download dataset**:
   ```bash
   # Option A: Web download
   # Click "Download" button for the complete archive

   # Option B: Command line
   wget https://zenodo.org/record/8003390/files/GRAPE_dataset.zip
   ```

3. **Extract and organize**:
   ```bash
   cd pde-paper/data/raw/
   mkdir -p grape
   unzip GRAPE_dataset.zip -d grape/
   ```

### Expected Structure
```
data/raw/grape/
├── fundus_images/           # Fundus photographs
├── visual_fields/           # Visual field test results
└── clinical_metadata.csv    # IOP measurements, patient demographics
```

---

## Data Processing

After downloading all datasets, run the image processing script to extract clinical parameters:

```bash
# Activate virtual environment
source venv/bin/activate

# Extract parameters from all datasets
python src/image_processing.py --extract-all --output data/processed/

# Or process individual datasets
python src/image_processing.py --dataset papila --output data/processed/papila/
python src/image_processing.py --dataset hygd --output data/processed/hygd/
python src/image_processing.py --dataset grape --output data/processed/grape/
```

This will generate:
- `data/processed/cdr_measurements.csv` - Cup-to-disc ratios from PAPILA
- `data/processed/iop_distributions.csv` - IOP statistics from GRAPE and HYGD
- `data/processed/geometry_parameters.json` - Anatomical measurements for simulations

---

## Verification

To verify successful dataset acquisition and processing:

```bash
# Check directory structure
ls -R data/raw/

# Verify image counts
find data/raw/hygd -name "*.jpg" | wc -l     # Should output: 747
find data/raw/papila -name "*.jpg" | wc -l   # Should output: 488
find data/raw/grape -name "*.jpg" | wc -l    # Should output: varies

# Test parameter extraction
python src/image_processing.py --verify
```

Expected output:
```
✓ HYGD: 747 images loaded
✓ PAPILA: 488 images, 244 patients (bilateral)
✓ GRAPE: 264 patients with longitudinal data
✓ Clinical parameters extracted successfully
```

---

## Disk Space Requirements

- **HYGD**: ~500 MB
- **PAPILA**: ~3 GB
- **GRAPE**: ~1.5 GB
- **Total**: ~5 GB

Ensure you have sufficient storage before downloading.

---

## License Compliance

All three datasets are licensed under **CC BY 4.0** (Creative Commons Attribution 4.0 International).

**You are free to**:
- Share and redistribute
- Adapt and build upon the material

**Under the following terms**:
- **Attribution**: You must cite the original dataset papers (see citations above)
- **No additional restrictions**: You may not apply legal terms or technological measures that restrict others

When publishing work using these datasets, include proper citations in your references.

---

## Troubleshooting

### PhysioNet Access Issues
- **Problem**: Cannot download HYGD dataset
- **Solution**: Ensure you have completed PhysioNet credentialing and signed the data use agreement

### Large File Downloads
- **Problem**: Download times out or fails
- **Solution**: Use command-line tools (`wget` with `-c` flag to resume) or download individual files

### Extraction Errors
- **Problem**: `unzip` command fails
- **Solution**: Ensure you have enough disk space and try extracting to a different location

### Missing Files
- **Problem**: Expected files not found after extraction
- **Solution**: Check the dataset version and compare with the original repository structure

---

## Contact

For dataset-specific questions:
- **HYGD**: Contact PhysioNet support or paper authors
- **PAPILA**: Open issue on Figshare or contact corresponding author
- **GRAPE**: Contact Zenodo repository maintainers

For questions about data processing in this repository:
- Open a GitHub issue in this repository

---

**Last Updated**: April 2026
