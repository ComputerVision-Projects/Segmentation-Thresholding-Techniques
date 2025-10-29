# PART A: Segmentation Studio

## Overview
This desktop application provides a comprehensive platform for experimenting with various **image segmentation techniques** — both classical and modern — with an intuitive user interface and modular class-based implementation. It enables users to visualize, compare, and analyze the behavior of segmentation algorithms like **Thresholding, Mean Shift, K-Means, Agglomerative Clustering, and Region Growing**.  

The tool is built using **Python (PyQt5 + OpenCV)** and designed for educational use in digital image processing, providing both interactive parameter control and side-by-side visual comparison of segmentation results.

---

## Screenshots

### Main Interface
![Main Interface](assets/main_interface.png)

### Segmentation Examples
![Segmentation Results](assets/segmentation_results.png)

---

## Features

### 1. Input Image Section
- **Display & Upload**: Supports color or grayscale image input.  
- **Upload Button**: Opens a file dialog for image selection.  
- **Auto Conversion**: Converts color images to grayscale where required.  

### 2. Algorithm Selection
Users can select from multiple segmentation algorithms:
- **Mean Shift Segmentation**
- **Agglomerative Clustering**
- **K-Means Clustering**
- **Region Growing**

### 3. Parameter Controls
Dynamic control panels appear based on the selected algorithm:

#### K-Means
- Number of clusters (slider or spinbox)

#### Mean Shift
- Spatial radius  
- Color radius  
- Bandwidth  
- Merging threshold  
- Optional gradient inclusion

#### Region Growing
- Seed point selection (click on image)  
- Similarity threshold slider  
- Minimum region size  

#### Agglomerative Clustering
- Number of clusters  
- Initial number of clusters for coarse grouping

- **"Apply" Button**: Executes segmentation with current settings.

### 4. Output Display
- **Dual View Panels**:  
  - Left: Original input image  
  - Right: Segmented output
- **Visualization Options**:  
  - Show boundaries  
  - Apply smoothing  
  - View cluster centers  

---

# PART B: Class Implementations

The system is divided into **7 modular classes** for scalability, reusability, and clarity.

## 1. Thresholder Class
Implements global and local thresholding algorithms:
- Optimal Global & Local Thresholding  
- Otsu’s Global & Local Methods  
- Spectral (Three-Level) Thresholding

**Key Methods:**
- `optimal_global(self, margin=10, convergence_threshold=0.5)`  
- `optimal_local(self, block_size=80, margin=3)`  
- `otsu_global(self)`  
- `otsu_local(self, block_size=50)`  
- `spectral_threshold(self)`  

> **Note:** Global thresholding methods produced more stable and reliable results compared to local approaches.

---

## 2. MeanShift Class
Performs **non-parametric clustering** using joint spatial-color feature space.

**Key Methods:**
- `compute_features(image)`  
- `segment(image)`  

Supports spatial, color, and optional gradient features. Adjustable kernel bandwidth, spatial radius, and merging threshold.

---

## 3. Agglomerative Clustering Class
Implements **hierarchical clustering** for image segmentation.

**Workflow:**
1. Start with each pixel as a cluster  
2. Iteratively merge closest clusters  
3. Update cluster centroids  

**Analysis:**  
- 5 clusters → overgeneralized segmentation  
- 15 clusters → clearer boundaries, better accuracy

---

## 4. RegionGrowing Class
Performs **seed-based segmentation** by expanding a region according to pixel similarity.

**Parameters:**
- Threshold (pixel similarity)  
- Connectivity (4- or 8-neighbor)

**Example Results:**
- Threshold = 10 → limited segmentation  
- Threshold = 25 → more accurate object extraction

---

## 5. KMeansCluster Class
Performs **unsupervised color-based segmentation** using K-Means.

**Workflow:**
1. Initialize centroids  
2. Assign pixels to nearest cluster  
3. Update centroids iteratively  
4. Recreate segmented image

**Results:**
- 9 clusters → simplified regions  
- 17 clusters → closer to original image

---

# PART C: Results & Observations

| Algorithm | Type | Key Strength | Limitation |
|-----------|------|--------------|------------|
| Otsu / Optimal Thresholding | Global | Simple, fast | Not ideal for textured images |
| Mean Shift | Clustering | Preserves edges, smooth segmentation | Computationally expensive |
| Agglomerative | Hierarchical | Fine-grained control | Slower for large images |
| Region Growing | Local | Great for uniform intensity | Sensitive to seed & threshold |
| K-Means | Clustering | Clear color-based regions | May ignore spatial context |

---

## Required Libraries
```bash
PyQt5
numpy
opencv-python
Pillow
matplotlib
logging
sys
os
