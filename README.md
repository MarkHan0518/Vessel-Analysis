# Vessel-Analysis

This repository contains code that is useful for segmenting (non-deep learning-based), extracting features from, and quantifying large three-dimensional blood vessel datasets.

## Vessel Segmentation

For small 3D datasets (MB level), use `VesSegmentation/AllInOne/Vessel_seg.py`. It contains all the code needed for the vessel segmentation pipeline.

For large 3D datasets (GB level), the available RAM may run out. In such cases, we recommend performing each step separately using `VesSegmentation/StepByStep`.

## Feature Extraction & Quantification

Use `VesQuantification/Ves_Quant.py` for blood vessel feature extraction and quantifications. 

## Sample Dataset

The `SampleData` folder contains a small 3D blood vessel dataset. It represents a part of the entire mouse brain captured by a
customized light-sheet microscopy developed in the Biophotonic Molecular Laboratory at the University of Washington.
Feel free to try the entire pipeline discussed previously on this dataset!