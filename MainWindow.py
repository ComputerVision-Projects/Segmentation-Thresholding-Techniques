from PyQt5.QtWidgets import QMainWindow,QComboBox,QTabWidget, QSpinBox, QWidget, QApplication, QPushButton, QLabel, QSlider,QProgressBar,QCheckBox,QMessageBox
from PyQt5.QtGui import QIcon
import os
import sys
import cv2
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from ImageViewer import ImageViewer
import cv2
import numpy as np
import time
from MeanShift import MeanShift
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        loadUi("MainWindow.ui", self)



        self.input_image_segment = self.findChild(QWidget, "inputImage")
        self.output_image_segment = self.findChild(QWidget, "outputImage")

        self.apply_segmentation= self.findChild(QPushButton, "applySegmentation")
        self.select_method_for_segmentation = self.findChild(QComboBox, "segmentationCombo")

        self.input_viewer_segment = ImageViewer(input_view=self.input_image_segment, mode=True)
        self.output_viewer_segment = ImageViewer(output_view=self.output_image_segment, mode=True)

        self.spatialSlider = self.findChild(QSlider, "spatialSlider")
        self.mergingSlider = self.findChild(QSlider, "mergingSlider")
        self.bandwidthSlider = self.findChild(QSlider, "bandwidthSlider")
        self.colorSlider = self.findChild(QSlider, "colorSlider")
        self.gradientSlider = self.findChild(QSlider, "gradientSlider")

        self.spatialSlider.setRange(1, 50)
        self.colorSlider.setRange(1, 50)
        self.mergingSlider.setRange(10, 200)
        self.bandwidthSlider.setRange(5, 50)
        self.gradientSlider.setRange(1, 20)

        self.apply_segmentation.clicked.connect(self.apply_segmentation_clicked)




    def apply_segmentation_clicked(self):
        selected_method = self.select_method_for_segmentation.currentText()

        if selected_method == "Mean Shift":
            # Initialize viewers only if needed
            if self.input_viewer_segment is None:
                self.input_viewer_segment = ImageViewer(input_view=self.input_image_segment, mode=True)
            if self.output_viewer_segment is None:
                self.output_viewer_segment = ImageViewer(output_view=self.output_image_segment, mode=True)

            self.run_mean_shift()



        elif selected_method == "Agglomerative":
         if self.input_viewer_segment is None:
            self.input_viewer_segment = ImageViewer(input_view=self.input_image_segment, mode=True)
         if self.output_viewer_segment is None:
            self.output_viewer_segment = ImageViewer(output_view=self.output_image_segment, mode=True)

        # self.run_agglomerative()

        elif selected_method == "Region Growing":
         if self.input_viewer_segment is None:
            self.input_viewer_segment = ImageViewer(input_view=self.input_image_segment, mode=True)
         if self.output_viewer_segment is None:
            self.output_viewer_segment = ImageViewer(output_view=self.output_image_segment, mode=True)

        # self.run_region_growing()

        elif selected_method == "K-means":
         if self.input_viewer_segment is None:
            self.input_viewer_segment = ImageViewer(input_view=self.input_image_segment, mode=True)
         if self.output_viewer_segment is None:
            self.output_viewer_segment = ImageViewer(output_view=self.output_image_segment, mode=True)

        # self.run_kmeans()

        else:
         print("Unknown segmentation method selected.")

    def run_mean_shift(self):
        img = self.input_viewer_segment.get_loaded_image()
        if img is None:
            print("No input image loaded.")
            return

        spatial_radius = self.spatialSlider.value()
        color_radius = self.colorSlider.value()
        merging_threshold = self.mergingSlider.value() / 100.0
        bandwidth = self.bandwidthSlider.value() / 10.0
        gradient_radius = self.gradientSlider.value()

        add_gradients = gradient_radius > 0

        segmenter = MeanShift(
            spatial_radius=spatial_radius,
            color_radius=color_radius,
            merging_threshold=merging_threshold,
            bandwidth=bandwidth,
            add_gradients=add_gradients,
            gradient_radius=gradient_radius
        )

        segmented_image, _ = segmenter.segment(img)

        if segmented_image is not None:
            self.output_viewer_segment.display_output_image(segmented_image)
        else:
            print("Segmentation returned None.")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())        
    