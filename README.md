# 3D-iseqPLA: Spatiotemporal Immune Inflammation modulates 3D NFκB signaling interactomics of multiprotein supercomplexes

[![DOI](https://img.shields.io/badge/DOI-Preprint-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

**Nicholas Zhang<sup>1,2,3</sup>, Collin Leese-Thompson<sup>4,5</sup>, Sriya Sirigireddy<sup>1,3</sup>, Dhruv Nambiar<sup>1,3</sup>, Lakshana Ramanan<sup>1,3</sup>, Rabindra Tirouvanziam<sup>4,5</sup>, and Ahmet F. Coskun<sup>1,2,3,*</sup>**

<sup>1</sup> Wallace H. Coulter Department of Biomedical Engineering, Georgia Institute of Technology and Emory University, Atlanta, GA, USA
<sup>2</sup> Interdisciplinary Bioengineering Graduate Program, Georgia Institute of Technology, Atlanta, GA, USA
<sup>3</sup> Parker H. Petit Institute for Bioengineering and Bioscience, Georgia Institute of Technology, Atlanta, GA, USA
<sup>4</sup> Department of Pediatrics, Emory University, Atlanta, GA, USA
<sup>5</sup> Center for CF & Airways Disease Research, Children's Healthcare of Atlanta, Atlanta, GA, USA
<sup>*</sup> Corresponding author: ahmet.coskun@bme.gatech.edu

---

## Overview

This repository contains code, data, and analysis pipelines for the first **volumetric, in situ profiling of endogenous NFκB protein-protein interactions (PPIs)** using **iterative sequential proximity ligation assay (iseqPLA)** combined with spinning disk confocal microscopy and 3D reconstruction.

### Key Features

- **3D spatial interactomics** of NFκB signaling supercomplexes at single-cell resolution
- **iseqPLA workflow** for multiplexed PPI detection across sequential imaging cycles
- **PRISMS-based 3D reconstruction** pipeline for volumetric quantification
- **~50,000 cells** imaged across multiple experimental conditions
- **scGPT foundation model** validation of NFκB gene panel relevance
- Analysis of **cystic fibrosis patient-derived macrophages** in coculture systems

---

## Abstract

The NFκB signaling pathway orchestrates inflammatory responses through the dynamic assembly and dissociation of membrane-proximal multiprotein supercomplexes, yet their spatiotemporal organization within the three-dimensional (3D) intracellular space has remained unresolved at single-cell resolution. Here, we present the first volumetric, in situ profiling of endogenous NFκB protein-protein interactions (PPIs) using iterative sequential proximity ligation assay (iseqPLA) combined with spinning disk confocal microscopy and 3D reconstruction.

Across 01-3T3 mouse fibroblasts, IMR-90 human fibroblasts, and cystic fibrosis patient-derived macrophage cocultures, we characterize supercomplex dissociation kinetics, p65 nuclear translocation dynamics, and negative feedback engagement over a 105-minute cytokine time course. We demonstrate that:

- **3D volumetric quantification** resolves PPI distributions obscured by conventional 2D maximum intensity projections
- **Extracellular matrix coating** critically determines the fraction of NFκB-responsive cells
- **CF airway-conditioned macrophages** amplify paracrine NFκB signaling in adjacent fibroblasts

A transfer learning-based scGPT foundation model, trained on curated in vitro and in vivo transcriptomic datasets, confirms statistically significant enrichment of our selected NFκB gene panel within inflammation-relevant transcriptional feature space.

---

## Repository Structure

```
3D-iseqPLA/
├── code/
│   ├── image_processing/          # 3D confocal image processing scripts
│   ├── iseqPLA_analysis/          # PPI quantification and analysis
│   ├── foundation_model/          # scGPT training and evaluation
│   ├── visualization/             # Figure generation scripts
│   └── utils/                     # Helper functions
├── data/
│   ├── raw/                       # Raw microscopy images (not included - see Data Availability)
│   ├── processed/                 # Processed single-cell measurements
│   ├── transcriptomics/           # scRNA-seq datasets for foundation model
│   └── metadata/                  # Experimental metadata
├── figures/                       # Publication-quality figures
├── notebooks/                     # Jupyter notebooks for analysis
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for foundation model training)
- Minimum 32GB RAM (64GB recommended for large image processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/coskun-lab/3D-iseqPLA.git
cd 3D-iseqPLA

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`
- `opencv-python>=4.5.0`
- `scikit-image>=0.18.0`
- `napari>=0.4.0` (for 3D visualization)
- `torch>=1.10.0` (for scGPT)
- `scanpy>=1.8.0` (for scRNA-seq analysis)
- `scgpt` (foundation model)

---

## Experimental Design

### Cell Models

1. **01-3T3 mouse fibroblasts** (n=2,780 cells)
   - TNFα (10 ng/mL), IL-1β (1 ng/mL), DMSO control
   - Time course: 0–105 min in 15-min intervals

2. **IMR-90 human fibroblasts** (n=1,425 cells)
   - Upstream supercomplex panel: TRAF-5_TRADD, TRAF-5_TRAF-2
   - Same cytokine conditions

3. **CCL2 macrophage + IMR-90 cocultures** (n=16,961 cells)
   - Control condition (CCL2 chemokine attractant)
   - LPS (10 ng/mL), TNFα, IL-1β, DMSO
   - Time course: 0, 30, 60, 120, 240, 480 min

4. **CFASN macrophage + IMR-90 cocultures** (n=15,617 cells)
   - CF airway supernatant-conditioned macrophages
   - Same stimulation conditions

5. **IMR-90 monocultures** (n=13,221 cells)
   - Baseline comparison without macrophages

### iseqPLA Panel

**Cycle 1:** Reporter proteins (H2B, p65 protein)
**Cycle 2:** p105/p50 & p65 heterodimer
**Cycle 3:** A20 & IKKβ (negative feedback)
**Cycle 4:** A20 & IKKγ (negative feedback)

**Upstream panel:** TRAF-5_TRADD, TRAF-5_TRAF-2

### Imaging Parameters

- **Microscope:** Cephla Squid spinning disk confocal
- **Objective:** Nikon 60× water lens
- **Z-spacing:** 0.5 μm
- **Z-planes:** 40 per field of view
- **Channels:** DAPI, A488/GFP, ds-RED/TRITC, Cy5/647 nm
- **Stitching:** 3×3 grid per FOV

---

## Usage

### 1. Image Processing and 3D Reconstruction

Process raw confocal z-stacks into 3D volumetric renderings:

```bash
python code/image_processing/reconstruct_3d.py \
    --input data/raw/experiment_01/ \
    --output data/processed/3d_renderings/ \
    --z-spacing 0.5 \
    --num-planes 40
```

### 2. PPI Quantification

Quantify PPI puncta from iseqPLA images:

```bash
python code/iseqPLA_analysis/quantify_ppis.py \
    --input data/processed/3d_renderings/ \
    --output data/processed/ppi_measurements/ \
    --panel upstream  # Options: upstream, feedback, reporters
```

### 3. Single-Cell Analysis

Extract single-cell features and generate quantitative metrics:

```bash
python code/iseqPLA_analysis/single_cell_analysis.py \
    --input data/processed/ppi_measurements/ \
    --output data/processed/single_cell_features.csv \
    --compute-nc-ratio  # Nuclear-to-cytoplasmic p65 ratio
```

### 4. Foundation Model Training

Train scGPT model on curated transcriptomic datasets:

```bash
# In vitro training
python code/foundation_model/train_invitro.py \
    --data data/transcriptomics/invitro_datasets.h5ad \
    --output models/scgpt_invitro/ \
    --epochs 20 \
    --batch-size 32

# In vivo fine-tuning
python code/foundation_model/train_invivo.py \
    --pretrained models/scgpt_invitro/best_model.pt \
    --data data/transcriptomics/invivo_datasets.h5ad \
    --output models/scgpt_invivo/ \
    --epochs 10
```

### 5. Generate Figures

Reproduce publication figures:

```bash
python code/visualization/generate_all_figures.py \
    --data data/processed/ \
    --output figures/ \
    --format pdf
```

---

## Key Results

### 1. 3D vs 2D Quantification

3D volumetric analysis provides:
- **Reduced variance** in nuclear-to-cytoplasmic p65 ratios
- **More accurate** discrimination of nuclear vs. cytoplasmic PPIs
- **Elimination of artifacts** from z-plane signal overlap

**Example:** 2D analysis yielded N/C ratios ~3 AU at peak activation vs. ~2.5 AU in 3D, with systematically higher variance.

### 2. Supercomplex Dissociation Kinetics

- **TNFα** drives the most rapid and complete dissociation of TRAF-5_TRADD and TRAF-5_TRAF-2 supercomplexes
- **Dissociation begins at 45 min** and approaches baseline by 90–105 min
- **IL-1β** produces attenuated dissociation, consistent with distinct TNFR1 vs. IL-1R signaling architectures
- **Peak p65 activation** (30–45 min) coincides with maximal supercomplex dissociation

### 3. ECM Coating Effects

Substrate coating profoundly affects NFκB activation:

| Coating | TNFα-activated cells | IL-1β-activated cells |
|---------|---------------------|-----------------------|
| **Collagen I** | 87.5% (*** p<0.001) | 87.6% (ns) |
| **Poly-L-lysine** | 67.2% | 81.8% |
| **Matrigel** | 18.2% (*** p<0.001) | 29.8% |

**Implication:** Matrigel dramatically suppresses cytokine-induced NFκB activation, likely through cytokine sequestration.

### 4. CF Macrophage Paracrine Amplification

CFASN-exposed macrophages show:
- **Elevated IL-1β-induced p65 activation** (0.47 at 120 min vs. 0.25 for CCL2)
- **Sustained LPS response** with secondary elevation at 240–480 min
- **Hyperinflammatory phenotype** transmitted to adjacent fibroblasts via paracrine signaling

### 5. Foundation Model Validation

scGPT model confirms NFκB gene panel relevance:
- **In vitro model:** Mann-Whitney U test p = 0.0295* (significant enrichment)
- **In vivo model:** Non-significant (p = 0.264) due to greater transcriptomic heterogeneity
- **Model performance:** Accuracy, precision, recall, F1, AUC all >0.95

---

## Data Availability

Due to the extremely large size of raw microscopy datasets (>2 TB), raw images are available upon request from the corresponding author (ahmet.coskun@bme.gatech.edu).

**Processed data included in this repository:**
- Single-cell PPI measurements (CSV format)
- 3D volumetric features
- Foundation model training datasets (GEO accessions listed in Supplementary Tables 1–2)

**External datasets used:**
- In vitro: GSE94383, GSE199404, GSE189062, GSE132791, GSE197031, GSE120000, GSE226488 (n=243,268 cells)
- In vivo: 16 lung disease cohorts (COVID-19, IPF, CF, COPD, asthma, tuberculosis; n=10,000 cells)

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{zhang2025spatiotemporal,
  title={Spatiotemporal Immune Inflammation modulates 3D NFκB signaling interactomics of multiprotein supercomplexes},
  author={Zhang, Nicholas and Leese-Thompson, Collin and Sirigireddy, Sriya and Nambiar, Dhruv and Ramanan, Lakshana and Tirouvanziam, Rabindra and Coskun, Ahmet F.},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/XXXX.XX.XX.XXXXXX}
}
```

---

## Funding

This work was supported by:
- Lung Spore and the National Cancer Institute (P50CA217691)
- National Institutes of Health (R35GM151028, 1R33CA291197)
- Winship Cancer Institute of Emory University (P30CA138292)

---

## Contact

**Ahmet F. Coskun, Ph.D.**
Associate Professor
Wallace H. Coulter Department of Biomedical Engineering
Georgia Institute of Technology and Emory University
Email: ahmet.coskun@bme.gatech.edu
Lab Website: [coskun.gatech.edu](https://coskun.gatech.edu)

**Nicholas Zhang**
PhD Candidate
Interdisciplinary Bioengineering Graduate Program
Georgia Institute of Technology
Email: nzhang326@gatech.edu

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

We thank:
- Dr. Rabindra Tirouvanziam and the Center for CF & Airways Disease Research for providing CF patient-derived macrophages
- The Winship Cancer Institute Shared Resources for imaging support
- The scGPT development team for the foundation model framework

---

## Keywords

spatial interactomics, NFκB signaling, supercomplexes, 3D confocal, spatiotemporal dynamics, proximity ligation assay, cystic fibrosis, inflammation, single-cell analysis, foundation models
