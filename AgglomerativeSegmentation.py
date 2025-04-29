import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave 

class AgglomerativeClustering:
    def __init__(self, number_of_clusters, initial_number_of_clusters):
        # Number of final clusters to reduce to
        self.clusters_number = number_of_clusters
        # Number of initial clusters to start with
        self.initial_k = initial_number_of_clusters
        # List of clusters (each is a list of pixels)
        self.clusters_list = []
        # Dictionary mapping each pixel to its cluster number
        self.cluster = {}
        # Dictionary mapping each cluster number to its center (average color)
        self.centers = {}

    def calculate_distance(self, x1, x2):
        # Euclidean distance between two RGB points
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def clusters_average_distance(self, cluster1, cluster2):
        # Calculate average distance between two clusters by comparing their centers
        cluster1_center = np.average(cluster1, axis=0)
        cluster2_center = np.average(cluster2, axis=0)
        return self.calculate_distance(cluster1_center, cluster2_center)

    def initial_clusters(self, image_clusters):
        # Group pixels into initial_k rough clusters based on closest grayscale value
        groups = {}
        cluster_color = int(256 / self.initial_k)  # interval step for initial colors
        for i in range(self.initial_k):
            color = i * cluster_color
            groups[(color, color, color)] = []  # Use grayscale tuples as keys

        # Assign each pixel to the closest grayscale cluster center
        for p in image_clusters:
            go = min(groups.keys(), key=lambda c: np.sqrt(np.sum((p - c) ** 2)))
            groups[go].append(p)

        # Return non-empty groups as the initial clusters
        return [group for group in groups.values() if len(group) > 0]

    def get_cluster_center(self, point):
        # Return the center (average RGB value) of the cluster a point belongs to
        point_cluster_num = self.cluster[tuple(point)]
        return self.centers[point_cluster_num]

    def get_clusters(self, image_clusters):
        # Step 1: create initial clusters
        self.clusters_list = self.initial_clusters(image_clusters)

        # Step 2: merge clusters until desired number is reached
        while len(self.clusters_list) > self.clusters_number:
            # Find the pair of clusters with the smallest distance
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                key=lambda c: self.clusters_average_distance(c[0], c[1]))

            # Remove the merged clusters from the list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]
            # Merge and append the new cluster
            merged_cluster = cluster1 + cluster2
            self.clusters_list.append(merged_cluster)

        # Assign each point to a final cluster index
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        # Calculate and store the center of each final cluster
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def apply(self, image):
        # Resize image to reduce computation time
        resized_image = cv2.resize(image, (256, 256))
        # Flatten image to a list of RGB pixels
        flattened_image = np.copy(resized_image.reshape((-1, 3)))

        # Perform clustering on the flattened image
        self.get_clusters(flattened_image)

        # Build the output image using cluster centers
        output_image = []
        for row in resized_image:
            rows = [self.get_cluster_center(list(col)) for col in row]
            output_image.append(rows)

        # Convert the result to uint8 image format
        output_image = np.array(output_image, np.uint8)

        return output_image

