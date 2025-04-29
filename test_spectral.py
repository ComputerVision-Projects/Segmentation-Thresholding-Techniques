import matplotlib.pyplot as plt
from skimage import data, color
from SpectralThreshold import SpectralThreshold
# Load example image and convert to grayscale
image = color.rgb2gray(data.astronaut())

# Apply spectral thresholding
spectral = SpectralThreshold(image)
segmented = spectral.apply()

# Show results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(segmented, cmap='gray')
plt.title('Spectral Thresholded')
plt.show()
