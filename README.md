# Image Segmentation Application

## Overview

This desktop application is designed for experimenting with several **image segmentation techniques** including Thresholding, Mean Shift, K-Means, Agglomerative Clustering, and Region Growing. Users can visualize, manipulate, and analyze segmentation results interactively. The project provides a hands-on way to understand color-based, intensity-based, and region-based segmentation methods.

---

## Features

### **Input Image Section**
- **Open and View**: Load color or grayscale images.
  - **Auto Conversion**: Converts color images to grayscale when needed.
  - **File Dialog**: Easy image selection via browsing.
  - **Unified Sizes**: Ensures all loaded images have consistent display size.

### **Algorithm Selection**
- **K-Means Clustering**
- **Agglomerative Clustering**
- **Mean Shift**
- **Region Growing**

### **Parameter Controls**
- K-Means: Number of clusters
- Agglomerative: Number of clusters, initial clusters
- Mean Shift: Spatial radius, color radius, bandwidth, merging threshold, gradient option
- Region Growing: Seed selection, similarity threshold, connectivity

### **Output Display**
- Dual-view panels: Original vs. segmented image
- Option to show boundaries, cluster centers, or smoothing

---

## Segmentation Techniques

### **1. Thresholder Class**
Performs global and local thresholding:
- Optimal Thresholding
- Otsu Thresholding
- Spectral (three-level) Thresholding

**Example Output:**  
<img  width="780" height="449" alt="image" src="https://github.com/user-attachments/assets/5e1a65f1-98f4-450c-98f7-43b2f561c367" />
<img width="780" height="449" alt="image" src="https://github.com/user-attachments/assets/4bc29b21-0314-450d-9930-049caf328974" />

---
### **2. MeanShift Class**
Performs non-parametric clustering using spatial and color features.

---

### **3. Agglomerative Clustering Class**
Implements hierarchical clustering:
- Iteratively merges clusters based on color similarity.
- Centroids are updated dynamically.

**Example Output:**  
<img width="780" height="449" alt="image" src="https://github.com/user-attachments/assets/118e8ff4-c1bb-4dde-9395-cebcc923b333" />

---

### **4. RegionGrowing Class**
Seed-based segmentation that grows regions around a selected point:
- Pixel similarity threshold controls region expansion.
- Supports 4- or 8-connected neighborhood exploration.

**Example Output:**  
<img  width="780" height="449" alt="image" src="https://github.com/user-attachments/assets/df31d462-6d4c-403b-a2fc-8c5c8918c676" />


---

### **5. KMeansCluster Class**
Performs unsupervised color-based clustering:
- Initializes k centroids
- Assigns pixels to nearest centroid
- Updates centroids iteratively
- Reconstructs segmented image

**Example Output:**  
<img  width="780" height="449" alt="image" src="https://github.com/user-attachments/assets/026ffd40-26ea-4877-b80a-a17c51b60fa0" />


---

## Results & Observations

| Algorithm | Strength | Limitation |
|-----------|----------|------------|
| Thresholder | Simple, fast for uniform regions | Less effective on textured images |
| Mean Shift | Smooth clustering | Computationally intensive |
| Agglomerative | Fine-grained control | Slow on large images |
| Region Growing | Accurate for uniform regions | Sensitive to seed and threshold |
| K-Means | Captures color variations | Ignores spatial info |

---

### Required Libraries:
- `PyQt5`
- `numpy`
- `OpenCV`
- `Pillow`
- `Logging`
- `matplotlib`
- `sys`
- `os`

---

## Acknowledgments

Supervised by **Professor Ahmed Badwy** during the **Computer Vision** course at **Cairo University, Faculty of Engineering**.

![Cairo University Logo](https://imgur.com/Wk4nR0m.png)

---

## Contributors

- [Judy Essam](https://github.com/JudyEssam)  
- [Laila Khaled](https://github.com/LailaKhaled352)  
- [Fatma Elsharkawy](https://github.com/FatmaElsharkawy)  
- [Hajar Ehab](https://github.com/HajarEhab)
