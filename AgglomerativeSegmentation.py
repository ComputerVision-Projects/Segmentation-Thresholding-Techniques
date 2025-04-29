import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave 

class AgglomerativeClustering:
    def __init__(self, number_of_clusters, initial_number_of_clusters):
        self.clusters_number = number_of_clusters
        self.initial_k = initial_number_of_clusters
        self.clusters_list = []
        self.cluster = {}
        self.centers = {}

    def calculate_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def clusters_average_distance(self, cluster1, cluster2):
        cluster1_center = np.average(cluster1, axis=0)
        cluster2_center = np.average(cluster2, axis=0)
        return self.calculate_distance(cluster1_center, cluster2_center)

    def initial_clusters(self, image_clusters):
        groups = {}
        cluster_color = int(256 / self.initial_k)
        for i in range(self.initial_k):
            color = i * cluster_color
            groups[(color, color, color)] = []

        for p in image_clusters:
            go = min(groups.keys(), key=lambda c: np.sqrt(np.sum((p - c) ** 2)))
            groups[go].append(p)

        return [group for group in groups.values() if len(group) > 0]

    def get_cluster_center(self, point):
        point_cluster_num = self.cluster[tuple(point)]
        return self.centers[point_cluster_num]

    def get_clusters(self, image_clusters):
        self.clusters_list = self.initial_clusters(image_clusters)

        while len(self.clusters_list) > self.clusters_number:
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                key=lambda c: self.clusters_average_distance(c[0], c[1]))

            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]
            merged_cluster = cluster1 + cluster2
            self.clusters_list.append(merged_cluster)

        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def apply(self, image):
        resized_image = cv2.resize(image, (256, 256))
        flattened_image = np.copy(resized_image.reshape((-1, 3)))

        self.get_clusters(flattened_image)

        output_image = []
        for row in resized_image:
            rows = [self.get_cluster_center(list(col)) for col in row]
            output_image.append(rows)

        output_image = np.array(output_image, np.uint8)

        return output_image
