import numpy as np

class Thresholder:
    def __init__(self, image):
        self.image = image.astype(np.uint8)

    def optimal_global(self, margin=10, convergence_threshold=0.5):
        """
        global adaptive thresholding using optimal threshold
        
        Args:
            margin: margin for corner background estimation
            convergence_threshold: threshold for convergence
        """
        gray_image = self.image
        h, w = gray_image.shape

        # 1. Get corner pixels (assumed background)
        corner_mask = np.zeros_like(gray_image, dtype=bool)
        corner_mask[:margin, :margin] = True
        corner_mask[:margin, -margin:] = True
        corner_mask[-margin:, :margin] = True
        corner_mask[-margin:, -margin:] = True

        background_pixels = gray_image[corner_mask]
        object_pixels = gray_image[~corner_mask]

        # 2. Initialize threshold using background + object means
        mu_b = background_pixels.mean()
        mu_o = object_pixels.mean()
        T_old = 0
        T_new = (mu_b + mu_o) / 2.0

        # 3. Iterate to refine threshold
        while abs(T_new - T_old) > convergence_threshold:
            T_old = T_new

            background_mask = gray_image <= T_old
            object_mask = gray_image > T_old

            if np.sum(background_mask) == 0 or np.sum(object_mask) == 0:
                break  # avoid division by zero

            mu_b = gray_image[background_mask].mean()
            mu_o = gray_image[object_mask].mean()

            T_new = (mu_b + mu_o) / 2.0

        # 4. Threshold the image
        binary_image = np.where(gray_image > T_new, 255, 0).astype(np.uint8)
        return binary_image, T_new
    
        
    def optimal_local(self, block_size=90, margin=3, convergence_threshold=0.1):
        gray_image = self.image
        h, w = gray_image.shape
        binary_image = np.zeros_like(gray_image)
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                
                bh = min(block_size, h - y)
                bw = min(block_size, w - x)
                block = gray_image[y:y+bh, x:x+bw]
                
                if block.size == 0:
                    continue
                
                # Create corner mask
                corner_mask = np.zeros_like(block, dtype=bool)
                m = min(margin, bh//2, bw//2)  # Adaptive margin for small blocks
                
                if m > 0:  # Only proceed if we can get corners
                    corner_mask[:m, :m] = True
                    corner_mask[:m, -m:] = True
                    corner_mask[-m:, :m] = True
                    corner_mask[-m:, -m:] = True
                    
                    background_pixels = block[corner_mask]
                    mu_background = background_pixels.mean() if background_pixels.size > 0 else block.mean()
                else:
                    mu_background = block.mean()
                
                # Threshold calculation
                mu_object = block.mean()
                T_new = (mu_background + mu_object) / 2
                
                
                while True:
                    T_old = T_new
                    bg_mask = block <= T_old
                    fg_mask = ~bg_mask
                    
                    if bg_mask.sum() == 0 or fg_mask.sum() == 0:
                        break
                    
                    mu_background = block[bg_mask].mean()
                    mu_object = block[fg_mask].mean()
                    T_new = (mu_background + mu_object) / 2
                    
                    
                    if abs(T_new - T_old) <= convergence_threshold:
                        break
                
                # Apply threshold
                binary_image[y:y+bh, x:x+bw] = np.where(block > T_new, 255, 0).astype(np.uint8)
        
        return binary_image
    

    def otsu_global(self):
        """
        Applies Otsu's thresholding to the grayscale image.
        Returns:
            binary_image: thresholded image (0 or 255)
            threshold: optimal threshold value computed by Otsu
        """
        gray_image = self.image
        hist, _ = np.histogram(gray_image.ravel(), bins=256, range=(0, 256))
        total_pixels = gray_image.size

        prob = hist / total_pixels
        cumulative_sum = 0.0
        cumulative_mean = 0.0
        global_mean = np.dot(np.arange(256), prob)

        max_between_class_var = 0.0
        optimal_threshold = 0
        w0 = 0.0
        mu0 = 0.0

        for t in range(256):
            w0 += prob[t]
            if w0 == 0 or w0 == 1:
                continue  # skip thresholds with no foreground or background
            mu0 += t * prob[t]
            mu1 = (global_mean - mu0) / (1 - w0)
            between_class_var = w0 * (1 - w0) * ((mu0 / w0 - mu1) ** 2)

            if between_class_var > max_between_class_var:
                max_between_class_var = between_class_var
                optimal_threshold = t

        binary_image = np.where(gray_image > optimal_threshold, 255, 0).astype(np.uint8)
        return binary_image, optimal_threshold
    
    def otsu_local(self, block_size=50):
        """
        Apply Otsu thresholding locally in blocks.
        
        Args:
            block_size: size of the square block (e.g., 90x90)
            
        Returns:
            binary_image: thresholded binary image
        """
        gray_image = self.image
        h, w = gray_image.shape
        binary_image = np.zeros_like(gray_image, dtype=np.uint8)

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                bh = min(block_size, h - y)
                bw = min(block_size, w - x)
                block = gray_image[y:y+bh, x:x+bw]
                
                # Otsu threshold on the block
                hist, _ = np.histogram(block.ravel(), bins=256, range=(0, 256))
                total_pixels = block.size
                prob = hist / total_pixels
                global_mean = np.dot(np.arange(256), prob)

                w0 = 0.0
                mu0 = 0.0
                max_between_class_var = 0.0
                optimal_threshold = 0

                for t in range(256):
                    w0 += prob[t]
                    if w0 == 0 or w0 == 1:
                        continue
                    mu0 += t * prob[t]
                    mu1 = (global_mean - mu0) / (1 - w0)
                    between_class_var = w0 * (1 - w0) * ((mu0 / w0 - mu1) ** 2)

                    if between_class_var > max_between_class_var:
                        max_between_class_var = between_class_var
                        optimal_threshold = t
                
                # Apply local threshold
                binary_image[y:y+bh, x:x+bw] = np.where(block > optimal_threshold, 255, 0).astype(np.uint8)
        
        return binary_image
