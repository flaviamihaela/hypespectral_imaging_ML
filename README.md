
# hypespectral_imaging_ML

## Overview
This project contains a full Python ML pipeline for crop classification. The pipeline combines spectral imaging, texture analysis and ML based feature fusion to automatically discriminate between crop classes. 

For more information on spectral imaging and what it can tell us about chlorophyll content check the following <link>https://github.com/flaviamihaela/hypespectral_imaging_ML/wiki/Motivation-and-Additional-Information</link>.

<img width="791" height="257" alt="image" src="https://github.com/user-attachments/assets/0b982e7d-f460-4a07-80b2-2d0a78fca861" />

## Key Features

PART 1:
- Pre-processing the data: storing the data, reducing the noise and artifacts, and separating the leaves from background to retain the area of interest;

PART 2:
- Estimating texture parameters through a transform-based approach (Gabor magnitude features);
- Fusing the spectral features together with textural features in various combinations to discover the best multimodal framework for crop health monitoring;
- Classifying the images by using multivariate analysis (a modified version of a Support Vector Machine algorithm) on the hyperspectral leaf images;
- Evaluating the performance of serial fusion against the performance of the double serial fusion in the context of Multispectral agricultural applications;

## Datasets

PART 1:
- Light Leaf Spot (LLS) dataset used to prove the pre-processing, segmentation and normalisation steps at canopy level;

 PART 2:
- Indian Pines (IP) used to prove the compression (dimensionality reduction) and classification steps;

## PART 1 - Pre processing workflow - LLS dataset:

- Denoises every spectral band (Gaussian)

- Segments the leaf either with a universal edge detector (LoG) or vegetation‑index masks (NDVI/OSAVI)

- Extracts the mean reflectance per band inside the mask

- Smooths the resulting 1‑D spectrum using a Savitzky‑Golay filter
  
- The output is a noise‑robust, size‑reduced .csv spectrum ready for ML.


| Script                      | Core functionality                                                                                          | Outputs                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `ROI_extraction.py`         | Calls LoG pipeline end‑to‑end (Gaussian → LoG → SG) and saves averaged spectrum                        | `ROI_mask.png`, `leaf_spectra.csv`              |
| `log_extraction_average.py` | Stand‑alone **edge‑based pipeline** that creates the LoG mask and spectra                              | `mask.png`, `edges.png`, `spectra.csv`          |
| `ndvi_extraction.py`        | Generates an **NDVI mask** (`NDVI > thr`) then SG‑smooths the extracted spectra                        | `ndvi_mask.png`, `ndvi_spectra.csv`             |
| `osavi_extraction.py`       | Generates an **OSAVI mask** (`OSAVI > thr`) then SG‑smooths the extracted spectra                      | `osavi_mask.png`, `osavi_spectra.csv`           |

LoG extraction

<img width="196" height="255" alt="image" src="https://github.com/user-attachments/assets/fe1801ea-e86d-406b-ac79-c710b8bdb01c" />

Index extraction

<img width="198" height="259" alt="image" src="https://github.com/user-attachments/assets/65529cb5-b123-4f82-879a-76c162ff228d" />


Following link describes the vegetation indices in more detail: <link> https://github.com/flaviamihaela/hypespectral_imaging_ML/wiki/Vegetation-Indices </link>

## PART 2 - ML Pipeline - Indian Pines Dataset:

### Transforms the cube into a tabular ML set
- extract_pixels flattens every 145 × 145 pixel into a 1 × 200 vector and appends the ground-truth label; it is written once to Dataset.csv for reproducibility.

### Feature engineering blocks
- Spectral: PCA/LLE/KPCA reduce the 200-D spectra to 10–200 components, with the exact algorithm chosen via grid-search

- Textural: a 4-orientation Gabor filter bank is run on every spectral band; real & imag parts are scaled, squared-summed and square-rooted to obtain magnitude features

### Fusion strategies
1 .No Fusion
- Use a single feature stream.

- Spectral-only or Gabor-only features go straight into the SVM.

2. Serial Fusion — Independent (serial_fusion_independent.py)
- Compress first, then merge.

- Apply KPCA (or LLE) separately to the spectral and Gabor sets.

- Concatenate the two reduced vectors → feed to SVM.

3. Serial Fusion — Dependent (serial_fusion_dependent.py)
- Let spectra guide the textures.

- Reduce the spectral cube to a small set of PCs.

- Run the Gabor filter bank on those PCs.

- Concatenate PCs + Gabor features → SVM.

4. Double Serial Fusion (double_serial_fusion.py)
One more squeeze for good measure.

- Start with the independent serial fusion above.

- Apply a second KPCA to the concatenated vector to decorrelate and reduce it further before classification.

### Model selection & evaluation
- An RBF-SVM (20 % hold-out, stratified) is tuned with GridSearchCV over C and γ; metrics include overall accuracy, weighted F1 and Cohen’s κ, plus a Seaborn confusion-matrix heat-map saved as cmap.png (and, for the spectral baseline, a full scene classification map IP_cmap.png)

| Script                         | Core functionality                                                          | Typical outputs                                                       |
| ------------------------------ | --------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `no_fusion_spectral_fe_svm.py` | Spectral PCA → SVM baseline; also makes band/GT previews                    | `IP_Bands.png`, `IP_GT.png`, `Dataset.csv`, `cmap.png`, `IP_cmap.png` |
| `gabor_spectral_no_fusion.py`  | Either raw spectra **or** Gabor magnitude + LLE; no feature fusion          | `Dataset.csv`, `cmap.png`                                             |
| `serial_fusion_independent.py` | KPCA on spectra **and** KPCA on Gabor → concatenate → SVM                   | `Dataset.csv`, `cmap.png`, fused-feature CSV (in-memory)              |
| `serial_fusion_dependent.py`   | LLE compresses spectra → Gabor computed on first 10 PCs → concatenate → SVM | `Dataset.csv`, `cmap.png`                                             |
| `double_serial_fusion.py`      | Two-level fusion: (spectral + Gabor) → KPCA → (S + G + KPCA) → SVM          | `Dataset.csv`, `cmap.png`                                             |

## Conclusions

Experiments on the Indian Pines benchmark confirm that multimodal fusion outperforms single stream baselines:

Pipeline	Status	Typical Overall Accuracy

Spectral-only (PCA → SVM)	- 79 – 82 %

Gabor-only (LLE → SVM)	-	68 – 72 %

Serial Fusion — Independent	- 84 – 87 %

Serial Fusion — Dependent	-	81 – 84 %

Double Serial Fusion	- 80 %

[Exact scores vary with the number of components and the train/test split, but the ranking is consistent]

Independent serial fusion already delivers a sizeable boost by combining decorrelated spectral PCs with texture.

## Further improvements
- Insert a StandardScaler after every fusion step.
- Add random_state=42 globally for reproducibility.	Prevent silent fall-backs and make results repeatable.
- Switch from random pixel splits to blocked or field-wise cross-validation.	Removes spatial leakage + gives a truer estimate of generalisation.

[Note] : All the experiments were conducted on Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz, 1992 Mhz, 4 Core(s), 8 Logical Processor(s), using the experiment platform Spyder with Python version (3.8.). 
