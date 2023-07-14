# Vessel-Analysis

This repository contains code that is useful for segmenting (non-deep learning-based), extracting features from, and quantifying large three-dimensional blood vessel datasets.

## Vessel Segmentation

For small 3D datasets (MB level), use `VesSegmentation/AllInOne/Vessel_seg.py`. It contains all the code needed for the vessel segmentation pipeline, including multi-otsu thresholding, hysteresis thresholding, hole filling, and smoothing (in sequential order).

For large 3D datasets (GB level), the available RAM may run out. In such cases, we recommend performing each step separately using `VesSegmentation/StepByStep`.

### Post Processing after Vessel Segmentation

Artifacts may appear if there is too much black space in the dataset. These artifacts will only present in the surrounding black regions and should not interfere with the segmented blood vessels. However, for aesthetic and quantification purposes, we recommend applying the following steps:

1. Open "ImageJ"
2. Go to "Analyze" -> "Tools" -> "ROI Manager"
3. Use the ROI Manager to remove the artifacts in 3D dataset

For more details, refer to the `postprocessingtips.pdf` file.

## Feature Extraction

Use `VesQuantification/FeatureExtraction.py` for blood vessel feature extraction, including the centerline, bifurcations, and radius.

### Post Processing after Feature Extraction

We recommend following these steps for post-processing after feature extraction:

1. Open "ImageJ"
2. Go to "Analyze" -> "3D Object Counter"
3. Apply the "3D Object Counter" to the `bifurcations.tif` file
4. Save the center of mass of the bifurcation mask to a new file called `COM_bifurcations.tif`
5. Use `VesQuantification/BinaryConverter.py` to convert `COM_bifurcations.tif` into `COM_bifurcations_binary.tif`
6. The resulting bifurcation mask can more accurately represent the position and number of bifurcations.

For more details, refer to the `postprocessingtips.pdf` file.

## Feature Quantification

After extracting the useful features, run `VesQuantification/FeatureQuantification.py` to obtain the final quantifications.

## Sample Dataset

The `SampleData` folder contains a small 3D blood vessel dataset. It represents a part of the entire mouse brain captured by a
customized light-sheet microscopy developed in the Biophotonic Molecular Laboratory at the University of Washington.
Feel free to try the entire pipeline discussed previously on this dataset!