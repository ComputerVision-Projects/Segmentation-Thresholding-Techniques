from PyQt5.QtWidgets import QMainWindow,QComboBox,QTabWidget, QSpinBox, QWidget, QApplication, QPushButton, QLabel, QSlider,QProgressBar,QCheckBox,QMessageBox
from PyQt5.QtGui import QIcon
import os
import sys
import cv2
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi

import cv2
import numpy as np
from ImageViewer import ImageViewer
from KMeansCluster import KMeansCluster
from RegionGrowing import RegionGrowing

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        loadUi("MainWindow.ui", self)

        #kmeans&region growing tab
        self.input_widget_tab1 = self.findChild(QWidget, "inputImage")
        self.segmented_widget_tab1 = self.findChild(QWidget, "segmentedImage")
        self.input_viewer_tab1 = ImageViewer(input_view=self.input_widget_tab1, mode=True)
        self.segmented_viewer_tab1 = ImageViewer(output_view=self.segmented_widget_tab1, mode=True)    
        
        self.threshold_slider = self.findChild(QSlider, "thresholdSlider")
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(50)
        self.combox_segment_method=self.findChild(QComboBox, "segmentMethod")
        self.apply_segment= self.findChild(QPushButton, "apply_segment")
        self.apply_segment.clicked.connect(self.apply_segmentation_method)

    def apply_segmentation_method(self):
        index= self.combox_segment_method.getCurrentIndex()
        if index==0: #RegionGrowing
            self.apply_region_growing()
        else:
            self.apply_kmeans()

    def apply_kmeans(self):
        image = self.input_viewer_tab1.get_loaded_image()
        kmeans = KMeansCluster(image, k=self.threshold_slider.value())
        result = kmeans.apply_kmeans()
        self.segmented_viewer_tab1.display_output_image(result)

    def apply_region_growing(self):
        image = self.input_viewer_tab1.get_loaded_image()
        seed_point=self.input_viewer_tab1.get_seed_point()
        print(f"seed point: {seed_point}")
        self.region_grower = RegionGrowing(image) 
        result = self.region_grower.segment_image(seed_point, threshold=self.threshold_slider.value())
        self.segmented_viewer_tab1.display_output_image(result)
          
        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())        
    