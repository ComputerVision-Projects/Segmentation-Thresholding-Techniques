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

    def _get_neighbors(self, x, y, connectivity=4):
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
    def segment_image(self, seed_points, threshold):
        for seed in seed_points:
            self._grow_region(seed, threshold)
        return self.segmented_image


# class ImageCanvas(QWidget):
#     def __init__(self, output_widget):
#         super().__init__()  
#         # self.setFixedSize(self.pixmap.size())

#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             pos = event.pos()
#             if 0 <= pos.x() < self.pixmap.width() and 0 <= pos.y() < self.pixmap.height():
#                 self.seed_point = QPoint(pos.x(), pos.y())
#                 print(f"Selected Seed: ({self.seed_point.x()}, {self.seed_point.y()})")

#                 # Call region growing algorithm with this new seed
#                 self.region_grower = RegionGrowing(self.cv_image)  # reinitialize for fresh segmentation
#                 segmented = self.region_grower.segment_image([(self.seed_point.x(), self.seed_point.y())], threshold=30)
#                 self.segmented_img= self.numpy_to_qimage(segmented)
#                 pixmap = QPixmap.fromImage(self.segmented_img)
#                 self.output_widget.label.setPixmap(pixmap)

#                 self.show_mask = True
#                 self.update()  # force repaint
    
#     def numpy_to_qimage(self, np_img):
#         height, width = np_img.shape
#         bytes_per_line = width
#         q_image = QImage(np_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
#         return q_image

#     def paintEvent(self, event):
#         painter = QPainter(self)

#         if not self.show_mask:
#             painter.drawPixmap(0, 0, self.pixmap)

#         if self.seed_point and self.show_mask:
#             pen = QPen(Qt.red)
#             pen.setWidth(5)
#             painter.setPen(pen)
#             painter.drawEllipse(self.seed_point, 5, 5)  # draw seed point


# class OutputWidget(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.label = QLabel("Segmented mask will appear here")
#         layout = QHBoxLayout()
#         layout.addWidget(self.label)
#         self.setLayout(layout)

# class MainWindow(QWidget):
#     def __init__(self, image_path):
#         super().__init__()
#         self.setWindowTitle("Region Growing Segmentation")

#         self.output_widget = OutputWidget()
#         self.image_canvas = ImageCanvas(image_path, self.output_widget)

#         layout = QHBoxLayout()
#         layout.addWidget(self.image_canvas)
#         layout.addWidget(self.output_widget)

#         self.setLayout(layout)
#         self.resize(1200, 600)


# class OutputWidget(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.label = QLabel("Segmented mask will appear here")
#         layout = QHBoxLayout()
#         layout.addWidget(self.label)
#         self.setLayout(layout)

# class MainWindow(QWidget):
#     def __init__(self, image_path):
#         super().__init__()
#         self.setWindowTitle("Region Growing Segmentation")

#         self.output_widget = OutputWidget()
#         self.image_canvas = ImageCanvas(image_path, self.output_widget)

#         layout = QHBoxLayout()
#         layout.addWidget(self.image_canvas)
#         layout.addWidget(self.output_widget)

#         self.setLayout(layout)
#         self.resize(1200, 600)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     image_path ="D:\SBME 2026\(3rd year 2nd term) Sixth Term\computer vision\segment cancer.jpeg"  # Change this to a valid path
#     main_window = MainWindow(image_path)
#     main_window.show()
#     sys.exit(app.exec_())

