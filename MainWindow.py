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
from ImageViewer import ImageViewer
from KMeansCluster import KMeansCluster
from RegionGrowing import RegionGrowing
from Thresholder import Thresholder
from MeanShift import MeanShift

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        loadUi("MainWindow.ui", self)

        #thresholding
        self.input_image_threshold = self.findChild(QWidget, "inputImageTh")
        self.output_image_threshold = self.findChild(QWidget, "outputImageTh")

        self.input_viewer = ImageViewer(input_view=self.input_image_threshold, mode=False)
        self.output_viewer = ImageViewer(output_view=self.output_image_threshold, mode=False)

        self.thresholding_combobox = self.findChild(QComboBox, "thresholdingCombo")

        self.apply_global_thresholding = self.findChild(QPushButton, "applyGlobalThreshold")
        self.apply_local_thresholding = self.findChild(QPushButton, "applyLocalThreshold")

        self.apply_global_thresholding.clicked.connect(self.apply_global_threshold)
        self.apply_local_thresholding.clicked.connect(self.apply_local_threshold)



        #hajer/judy
        self.input_image_segment = self.findChild(QWidget, "inputImage")
        self.output_image_segment = self.findChild(QWidget, "outputImage")

        self.apply_segmentation= self.findChild(QPushButton, "applySegmentation_2")
        self.select_method_for_segmentation = self.findChild(QComboBox, "segmentationCombo_2")

        self.input_viewer_segment = ImageViewer(input_view=self.input_image_segment, mode=True)
        self.output_viewer_segment = ImageViewer(output_view=self.output_image_segment, mode=True)

        self.spatialSlider = self.findChild(QSlider, "spatialSlider_2")
        self.mergingSlider = self.findChild(QSlider, "mergingSlider_2")
        self.bandwidthSlider = self.findChild(QSlider, "bandwidthSlider_2")
        self.colorSlider = self.findChild(QSlider, "colorSlider_2")
        self.gradientSlider = self.findChild(QSlider, "gradientSlider_2")

        self.spatialLabel= self.findChild(QLabel, "spatialLabel_3")
        self.bandwidthLabel= self.findChild(QLabel, "bandwidthLabel_2")
        self.colorLabel= self.findChild(QLabel, "colorLabel_2")
        self.gradientLabel= self.findChild(QLabel, "gradientLabel_2")
        self.mergingLabel= self.findChild(QLabel, "mergingLabel_2")



        self.spatialSlider.setRange(1, 50)
        self.colorSlider.setRange(1, 50)
        self.mergingSlider.setRange(10, 200)
        self.bandwidthSlider.setRange(5, 50)
        self.gradientSlider.setRange(1, 20)
        self.spatialSlider.valueChanged.connect(self.update_slider_labels)
        self.bandwidthSlider.valueChanged.connect(self.update_slider_labels)
        self.colorSlider.valueChanged.connect(self.update_slider_labels)
        self.gradientSlider.valueChanged.connect(self.update_slider_labels)
        self.mergingSlider.valueChanged.connect(self.update_slider_labels)

        # Initial update so labels reflect default slider positions
        self.update_slider_labels()

        self.apply_segmentation.clicked.connect(self.apply_segmentation_clicked)

        #kmeans&region growing tab
        self.input_widget_tab1 = self.findChild(QWidget, "inputImage_2")
        self.segmented_widget_tab1 = self.findChild(QWidget, "segmentedImage_2")
        self.input_viewer_tab1 = ImageViewer(input_view=self.input_widget_tab1, mode=True)
        self.segmented_viewer_tab1 = ImageViewer(output_view=self.segmented_widget_tab1, mode=True)    
        
        self.threshold_slider = self.findChild(QSlider, "thresholdSlider")
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(50)
        self.combox_segment_method=self.findChild(QComboBox, "segmentMethod")
        self.apply_segment= self.findChild(QPushButton, "apply_segment")
        self.apply_segment.clicked.connect(self.apply_segmentation_method)

    def apply_global_threshold(self):
       method = self.thresholding_combobox.currentText()
       image = self.input_viewer.get_loaded_image()
       if image is None:
            print("No image loaded.")
            return
       # make sure it's grayscale
       gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
       thresholder = Thresholder(gray_image)
       if "Optimal" in method:
            binary_image, threshold_value = thresholder.optimal_global()
       elif "Otsu" in method:
            binary_image, optimal_threshold= thresholder.otsu_global()
    #    elif "Spectral" in method:
    #         binary_image = thresholder.spectral_global()
       else:
            return 

       self.output_viewer.display_output_image(binary_image)

    def apply_local_threshold(self):
       method = self.thresholding_combobox.currentText()
       image = self.input_viewer.get_loaded_image()
       if image is None:
            print("No image loaded.")
            return
       # make sure it's grayscale
       gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
       thresholder = Thresholder(gray_image)
       if "Optimal" in method:
            binary_image = thresholder.optimal_local()
       elif "Otsu" in method:
            binary_image = thresholder.otsu_local()
    #    elif "Spectral" in method:
    #         binary_image = thresholder.spectral_global()
       else:
            return 

       self.output_viewer.display_output_image(binary_image)


    def apply_segmentation_method(self):
        index= self.combox_segment_method.currentIndex()
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

    def update_slider_labels(self):
     self.spatialLabel.setText(f" {self.spatialSlider.value()}")
     self.bandwidthLabel.setText(f" {self.bandwidthSlider.value() / 10.0:.1f}")
     self.colorLabel.setText(f" {self.colorSlider.value()}")
     self.gradientLabel.setText(f" {self.gradientSlider.value()}")
     self.mergingLabel.setText(f" {self.mergingSlider.value() / 100.0:.2f}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())        
    