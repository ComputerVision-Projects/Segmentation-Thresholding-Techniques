# 🖼️ Segmentation Studio

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-yes-green.svg)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
  - [Input Image Section](#1-input-image-section)
  - [Algorithm Selection](#2-algorithm-selection)
  - [Parameter Controls](#3-parameter-controls)
  - [Output Display](#4-output-display)
- [Class Implementations](#class-implementations)
  - [Thresholder Class](#1-thresholder-class)
  - [MeanShift Class](#2-meanshift-class)
  - [Agglomerative Clustering Class](#3-agglomerative-clustering-class)
  - [RegionGrowing Class](#4-regiongrowing-class)
  - [KMeansCluster Class](#5-kmeanscluster-class)
- [Results & Observations](#results--observations)
- [Required Libraries](#required-libraries)
- [Acknowledgments](#acknowledgments)
- [Contributors](#contributors)

---

## Overview
**Segmentation Studio** is a desktop application for experimenting with **image segmentation techniques**:

- Thresholding (Optimal & Otsu)
- Mean Shift
- K-Means Clustering
- Agglomerative Clustering
- Region Growing

It provides a **modular, interactive interface** to visualize and analyze segmentation results, supporting **educational use** in digital image processing. Built with **Python, PyQt5, and OpenCV**.

---

## Features

### 1. Input Image Section
- **Upload & Display**: Supports color or grayscale images  
- **Auto Conversion**: Converts color images to grayscale where needed  
- **File Dialog**: Easy image selection with file browser  

### 2. Algorithm Selection
- Mean Shift  
- Agglomerative Clustering  
- K-Means Clustering  
- Region Growing  

### 3. Parameter Controls
Dynamic controls based on selected algorithm:

**K-Means**  
- Number of clusters (slider/spinbox)

**Mean Shift**  
- Spatial radius  
- Color radius  
- Bandwidth  
- Merging threshold  
- Optional gradient features  

**Region Growing**  
- Seed point selection  
- Similarity threshold  
- Minimum region size  

**Agglomerative Clustering**  
- Number of clusters  
- Initial cluster count  

**Apply Button** executes the selected algorithm.

### 4. Output Display
- **Dual View Panels**: Original vs. segmented output  
- **Visualization Options**: Show boundaries, smoothing, cluster centers  

---

## Class Implementations

### 1. Thresholder Class
Provides global and local thresholding methods:
- Optimal thresholding (global & local)  
- Otsu thresholding (global & local)  
- Spectral (three-level) thresholding  

**Key Methods:**  
```python
optimal_global(margin=10, convergence_threshold=0.5)
optimal_local(block_size=80, margin=3)
otsu_global()
otsu_local(block_size=50)
spectral_threshold()
