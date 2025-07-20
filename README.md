
# hypespectral_imaging_ML

## Overview
This project contains a full Python ML pipeline for crop classification. The pipeline combines spectral imaging, texture analysis and ML based feature fusion to automatically discriminate between crop classes. 

For more information on spectral imaging check the following <link>https://github.com/flaviamihaela/hypespectral_imaging_ML/wiki/Motivation-and-Additional-Information</link>.

<img width="791" height="257" alt="image" src="https://github.com/user-attachments/assets/0b982e7d-f460-4a07-80b2-2d0a78fca861" />

## Key Features

PART 1:
- Pre-processing the data: storing the data, reducing the noise and artifacts, and separating the leaves from background to retain the area of interest;

PART 2:
- Estimating texture parameters through a transform-based approach (Gabor magnitude features);
- Fusing the spectral features together with textural features in various combinations to discover the best multimodal framework for crop health monitoring;
- Classifying the images by using multivariate analysis (a modified version of a Support Vector Machine algorithm) on the hyperspectral leaf images;
- Evaluating the performance of serial fusion against the performance of the double serial fusion in the context of MS agricultural applications;

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







Following link describes the datasets in more detail: <link> https://github.com/flaviamihaela/hypespectral_imaging_ML/wiki/Datasets </link>

## Part 1 LLS dataset




[Note] : All the experiments were conducted on Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz, 1992 Mhz, 4 Core(s), 8 Logical Processor(s), using the experiment platform Spyder with Python version (3.8.). 
