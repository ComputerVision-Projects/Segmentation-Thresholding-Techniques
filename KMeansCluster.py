import numpy as np

class KMeansCluster:
    def __init__(self, image, k):
        '''
        Parameters:
        - image: input image (H x W x 3) in RGB or BGR format
        - k: number of clusters
        '''
        self.image = image
        self.k = k

    # Step 1: Initialize centroids using uniformly distributed values from image range
    def _initialize_points(self):
        pixels = self.image.reshape(-1, 3)
        min_vals = np.min(pixels, axis=0)
        max_vals = np.max(pixels, axis=0)
        
        # Uniformly initialize between min and max values per channel
        centroids = np.array([
            np.random.uniform(min_vals, max_vals)
            for _ in range(self.k)
        ], dtype=np.float32)
        return centroids

    # Step 2: Assign each pixel to the nearest centroid
    def assign_clusters(self, pixels, centroids):
        distances = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        return np.argmin(distances, axis=1)

    # Step 3: Update centroids
    def update_centroids(self, pixels, labels):
        new_centroids = np.zeros((self.k, 3), dtype=np.float32)
        for k in range(self.k):
            assigned = pixels[labels == k]
            if len(assigned) > 0:
                new_centroids[k] = np.mean(assigned, axis=0)
        return new_centroids

    def kmeans(self, max_iter=10):
        pixels = self.image.reshape(-1, 3).astype(np.float32)
        centroids = self._initialize_points()

        for _ in range(max_iter):
            labels = self.assign_clusters(pixels, centroids)
            new_centroids = self.update_centroids(pixels, labels)
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return centroids, labels

    def recreate_image(self, centroids, labels):
        image_shape = self.image.shape
        new_pixels = centroids[labels].astype(np.uint8)
        return new_pixels.reshape(image_shape)

    def apply_kmeans(self):
        centroids, labels = self.kmeans()
        return self.recreate_image(centroids, labels)
