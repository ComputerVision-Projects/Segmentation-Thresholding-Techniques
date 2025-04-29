import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
import sys

class RegionGrowing:
    def __init__(self, cl_image):
        self.cl_image = cl_image #original colored
        self.gr_image = cv2.cvtColor(cl_image, cv2.COLOR_BGR2GRAY) #gray image
        self.height, self.width = self.gr_image.shape
        self.segmented_image = np.zeros_like(self.gr_image)
        self.visited = np.zeros_like(self.gr_image, dtype=bool) #to save visited seeds

    #Step 2: growing the region around seedpoint. It useded helper methods to do its function: _get_neighbors & _check_similarity
    def _grow_region(self, seed_point, threshold):
        seed_x, seed_y = seed_point
        if seed_y > self.height or seed_x > self.width or self.visited[seed_y, seed_x]: #make sure seed doesn't exceed image dimensions
            return # Seed already visited

        region = [(seed_x, seed_y)]
        self.visited[seed_y, seed_x] = True
        boundary = self._get_neighbors(seed_x, seed_y)

        while boundary:
            current_x, current_y = boundary.pop(0) # Process boundary pixels FIFO

            if not self.visited[current_y, current_x] and self._check_similarity((current_x, current_y), seed_point, threshold):
                region.append((current_x, current_y))
                self.visited[current_y, current_x] = True
                neighbors = self._get_neighbors(current_x, current_y)
                for neighbor in neighbors:
                    if not self.visited[neighbor[1], neighbor[0]] and neighbor not in boundary:
                        boundary.append(neighbor)

        # Mark the grown region in the segmented image (you might want to assign a unique color or intensity)
        for pixel in region:
            self.segmented_image[pixel[1], pixel[0]] = 255 # Example: Mark region with white

        return region

    def _get_neighbors(self, x, y, connectivity=8):
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if connectivity == 4 and abs(dx) + abs(dy) == 1:
                        neighbors.append((nx, ny))
                    elif connectivity == 8:
                        neighbors.append((nx, ny))
        return neighbors

    def _check_similarity(self, pixel1, pixel2, threshold):
        x1, y1 = pixel1
        x2, y2 = pixel2
        return abs(int(self.gr_image[y1, x1]) - int(self.gr_image[y2, x2])) < threshold

    #called by mainwindow.(Step 1)
    def segment_image(self, seed_point, threshold):
        self._grow_region(seed_point, threshold)
        return self.segmented_image
