import numpy as np
import cv2
from scipy.spatial import KDTree

class MeanShift:
    def __init__(self, spatial_radius=10, color_radius=20, gradient_radius=5,
                 bandwidth=1.0, max_iter=5, eps=1e-3, merging_threshold=0.5,
                 add_gradients=False):
        self.spatial_radius = spatial_radius
        self.color_radius = color_radius
        self.gradient_radius = gradient_radius
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.eps = eps
        self.merging_threshold = merging_threshold
        self.add_gradients = add_gradients

    def gaussian_kernel(self, distance, bandwidth):
        return np.exp(-0.5 * (distance / bandwidth) ** 2)

    def compute_features(self, image):
        h, w, c = image.shape
        flat_image = np.reshape(image, (-1, 3))

        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        spatial = np.stack((yy.ravel(), xx.ravel()), axis=1)

        if self.add_gradients:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grads = np.stack((grad_x.ravel(), grad_y.ravel()), axis=1)
            feature_space = np.hstack((spatial, flat_image, grads))
        else:
            feature_space = np.hstack((spatial, flat_image))

        return feature_space.astype(np.float32)

    def segment(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w, _ = image.shape
        feature_space = self.compute_features(image)

        feature_space[:, 0:2] /= self.spatial_radius
        feature_space[:, 2:5] /= self.color_radius
        if self.add_gradients:
            feature_space[:, 5:7] /= self.gradient_radius

        shifted = np.copy(feature_space)
        active = np.ones((feature_space.shape[0],), dtype=bool)
        tree = KDTree(shifted)

        for iteration in range(self.max_iter):
            print(f"Iteration {iteration + 1}: active points = {np.sum(active)}")
            if np.sum(active) == 0:
                print("All points converged early!")
                break

            new_shifted = np.copy(shifted)
            for i in np.where(active)[0]:
                neighbors_idx = tree.query_ball_point(shifted[i], r=1.0)
                neighbors = shifted[neighbors_idx]
                distances = np.linalg.norm(neighbors - shifted[i], axis=1)
                weights = self.gaussian_kernel(distances, self.bandwidth)
                weighted_sum = np.sum(weights[:, np.newaxis] * neighbors, axis=0)
                new_pos = weighted_sum / np.sum(weights)
                shift_mag = np.linalg.norm(new_pos - shifted[i])

                if shift_mag < self.eps:
                    active[i] = False
                new_shifted[i] = new_pos

            shifted = new_shifted
            tree = KDTree(shifted)

        # Merging
        labels = -np.ones((feature_space.shape[0],), dtype=int)
        label = 0
        for i in range(feature_space.shape[0]):
            if labels[i] == -1:
                diff = shifted - shifted[i]
                distance = np.linalg.norm(diff, axis=1)
                neighbors = distance < self.merging_threshold
                labels[neighbors] = label
                label += 1

        print(f"Total clusters found: {label}")

        flat_image = np.reshape(image, (-1, 3))
        segmented_image = np.zeros((h * w, 3), dtype=np.uint8)
        for lbl in np.unique(labels):
            mask = labels == lbl
            cluster_color = np.mean(flat_image[mask], axis=0)
            segmented_image[mask] = cluster_color

        segmented_image = segmented_image.reshape((h, w, 3))
        return segmented_image, labels.reshape(h, w)
