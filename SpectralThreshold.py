import numpy as np
import cv2

import numpy as np
def spectral_threshold(img):
    #  Compute the histogram and global mean
    hist, _ = np.histogram(img, 256, [0, 256])
    mean = np.sum(np.arange(256) * hist) / float(img.size)
    #Search for optimal thresholds (low, high)
    optimal_high = 0
    optimal_low = 0
    max_variance = 0
    for high in range(0, 256):
        for low in range(0, high):
            #The pixels are split into 3 classes
            w0 = np.sum(hist[0:low])
            if w0 == 0:
                continue
            mean0 = np.sum(np.arange(0, low) * hist[0:low]) / float(w0)
            w1 = np.sum(hist[low:high])
            if w1 == 0:
                continue
            mean1 = np.sum(np.arange(low, high) * hist[low:high]) / float(w1)
            
            w2 = np.sum(hist[high:])
            if w2 == 0:
                continue
            mean2 = np.sum(np.arange(high, 256) * hist[high:]) / float(w2)
            # the class weights (w0, w1, w2) and class means (mean0, mean1, mean2 are calculated
            variance = w0 * (mean0 - mean) * 2 + w1 * (mean1 - mean) * 2 + w2 * (mean2 - mean) ** 2
            if variance > max_variance:
                max_variance = variance
                optimal_high = high
                optimal_low = low

    binary = np.zeros(img.shape, dtype=np.uint8)
    binary[img < optimal_low] = 0
    binary[(img >= optimal_low) & (img < optimal_high)] = 128
    binary[img >= optimal_high] = 255
    return binary

import matplotlib.pyplot as plt
image = cv2.imread("D:\SBME 2026\(3rd year 2nd term) Sixth Term\computer vision\segment cancer.jpeg", cv2.IMREAD_GRAYSCALE)
seg_result = spectral_threshold(image)

plt.imshow(seg_result, cmap='gray')
plt.title("Segmented Output")
plt.show()