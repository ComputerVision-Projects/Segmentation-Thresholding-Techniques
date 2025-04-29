import numpy as np
from scipy.sparse.linalg import eigsh
from skimage.transform import resize

class SpectralThreshold:
    def __init__(self, image, sigma=10):
        """
        Initialize the SpectralThreshold class.
        
        Parameters:
        image : 2D numpy array
            Input grayscale image.
        sigma : float
            Parameter to control the sensitivity of pixel similarity.
        """
        self.original_image = image
        self.image = resize(image, (30, 30))  # downsample for performance
        self.image = self.image / 255.0  # normalize pixel values
        self.sigma = sigma
        self.height, self.width = self.image.shape
        self.N = self.height * self.width  # total number of pixels (nodes)

    def build_affinity_matrix(self):
        """
        Construct the affinity (similarity) matrix W.
        W[i, j] = exp(-||I[i] - I[j]||^2 / 2σ^2)
        Only considers nearby pixels (e.g., within 8-connectivity).
        """
        flat_image = self.image.flatten()
        W = np.zeros((self.N, self.N), dtype=np.float32)

        for i in range(self.N):
            for j in range(i, self.N):
                # Only connect nearby pixels to reduce computation
                if abs(i - j) <= self.width + 1:
                    diff = flat_image[i] - flat_image[j]
                    W[i, j] = np.exp(-(diff**2) / (2 * self.sigma**2))
                    W[j, i] = W[i, j]  # Symmetric matrix
        return W

    def compute_laplacian(self, W):
        """
        Compute the unnormalized graph Laplacian L = D - W.
        D is the degree matrix (diagonal with row sums of W).
        """
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        return L

    def compute_fiedler_vector(self, L):
        """
        Compute the second smallest eigenvector of L (Fiedler vector).
        This vector is used for thresholding.
        """
        vals, vecs = eigsh(L, k=2, which='SM')  # smallest two eigenvalues
        return vecs[:, 1]  # Fiedler vector

    def threshold_vector(self, fiedler_vector):
        """
        Segment the image based on the Fiedler vector using zero threshold.
        """
        mask = fiedler_vector > 0
        return mask.reshape(self.image.shape)

    def apply(self):
        """
        Run the full spectral thresholding pipeline and return the mask.
        """
        W = self.build_affinity_matrix()
        L = self.compute_laplacian(W)
        fiedler_vector = self.compute_fiedler_vector(L)
        segmented_mask = self.threshold_vector(fiedler_vector)
        return segmented_mask
