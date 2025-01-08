import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import qr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from sklearn.metrics import mean_squared_error
#import psutil
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HolographicMemory:
    def __init__(self, dimensions=16384, initial_regularization=1e-6):
        """
        Initialize holographic memory with a high-dimensional space.
        :param dimensions: Number of dimensions for memory representation.
        :param initial_regularization: Starting regularization value for iterative encoding.
        """
        self.dimensions = dimensions
        self.memory_space = np.zeros(dimensions, dtype=complex)
        self.initial_regularization = initial_regularization
        self.keys = []
        self.values = []

    def normalize(self, vector):
        """Normalize a vector to unit length."""
        return vector / np.linalg.norm(vector)

    def encode(self, key, value, regularization):
        """
        Encode a key-value pair into holographic memory.
        :param key: Key vector (1D array).
        :param value: Value vector (1D array).
        :param regularization: Regularization factor to adjust encoding strength.
        """
        key = self.normalize(key)
        value = self.normalize(value)
        key_fft = fft(key, n=self.dimensions)
        value_fft = fft(value, n=self.dimensions)
        self.memory_space += key_fft * value_fft * (1 + regularization)

        # Store key-value pair for debugging or later use
        self.keys.append(key)
        self.values.append(value)

    def retrieve(self, key):
        """
        Retrieve the value associated with a given key.
        :param key: Input key vector (1D array).
        :return: Retrieved value vector (1D array).
        """
        key = self.normalize(key)
        key_fft = fft(key, n=self.dimensions)
        retrieved_fft = self.memory_space / (key_fft + 1e-9)  # Avoid division by zero
        retrieved_value = np.real(ifft(retrieved_fft))
        return self.noise_reduction(retrieved_value)

    def noise_reduction(self, value, gaussian_sigma=2, median_width=3):
        """
        Apply noise reduction to the retrieved value using Gaussian and median filters.
        :param value: Input vector (1D array).
        :param gaussian_sigma: Sigma for Gaussian filter.
        :param median_width: Kernel size for median filter.
        :return: Denoised value vector.
        """
        value = gaussian_filter1d(value, sigma=gaussian_sigma)
        value = medfilt(value, kernel_size=median_width)
        return value

    def compress_memory(self, threshold=None):
        """
        Compress the memory by removing low-magnitude elements.
        :param threshold: Threshold for compression. If None, use adaptive thresholding.
        """
        if threshold is None:
            threshold = self.adaptive_threshold()
        mask = np.abs(self.memory_space) > threshold
        self.memory_space[~mask] = 0

    def adaptive_threshold(self, percentile=90):
        """
        Calculate an adaptive threshold based on memory space values.
        :param percentile: Percentile for determining threshold.
        :return: Threshold value.
        """
        return np.percentile(np.abs(self.memory_space), percentile)

    def adaptive_regularization(self, iteration):
        """
        Calculate adaptive regularization for iterative encoding.
        :param iteration: Current iteration number.
        :return: Regularization value.
        """
        return self.initial_regularization * np.exp(-iteration / 10)

    def dynamic_encode(self, key, value, max_iterations=20, tolerance=1e-4):
        """
        Dynamically encode a key-value pair using adaptive regularization.
        :param key: Key vector (1D array).
        :param value: Value vector (1D array).
        :param max_iterations: Maximum number of encoding iterations.
        :param tolerance: Tolerance for convergence.
        """
        for i in range(max_iterations):
            regularization = self.adaptive_regularization(i)
            previous_memory = self.memory_space.copy()
            self.encode(key, value, regularization)
            if np.linalg.norm(self.memory_space - previous_memory) < tolerance:
                logging.info(f"[HolographicMemory] Converged after {i + 1} iterations.")
                break


#


def test_scaled_holographic_memory():
    dimensions = 16384  # Increased dimensions for high accuracy
    num_pairs = 100  # Number of key-value pairs
    memory = HolographicMemory(dimensions)

    logging.info(f"[Test] Encoding {num_pairs} key-value pairs into holographic memory...")
    start_time = time.time()

    # Generate orthogonal keys and random values
    keys = np.random.randn(dimensions, num_pairs)
    Q, _ = qr(keys, mode='economic')
    keys = Q.T
    values = [np.random.randn(dimensions) for _ in range(num_pairs)]

    # Encode key-value pairs with dynamic iterations
    for i in range(num_pairs):
        memory.dynamic_encode(keys[i], values[i])

    encoding_time = time.time() - start_time
    logging.info(f"[Performance] Encoding completed in {encoding_time:.2f} seconds.")

    logging.info("[Test] Retrieval and Accuracy Testing...")
    mse_list = []

    # Retrieve and calculate accuracy
    for i in range(num_pairs):
        retrieved_value = memory.retrieve(keys[i])
        mse = mean_squared_error(values[i], retrieved_value)
        mse_list.append(mse)

    average_mse = np.mean(mse_list)
    logging.info(f"\n[Results] Average MSE across {num_pairs} key-value pairs: {average_mse:.5f}")

    # Check if MSE is below the desired threshold
    if average_mse < 500:
        logging.info(f"MSE value is below 500: {average_mse:.5f}")
    else:
        logging.info(f"MSE value is above 500: {average_mse:.5f}")

    # Apply compression
    logging.info("\n[Compression Test] Applying compression to memory...")
    memory.compress_memory()

    mse_list_compressed = []

    # Retrieve and calculate accuracy after compression
    for i in range(num_pairs):
        retrieved_value_compressed = memory.retrieve(keys[i])
        mse_compressed = mean_squared_error(values[i], retrieved_value_compressed)
        mse_list_compressed.append(mse_compressed)

    average_mse_compressed = np.mean(mse_list_compressed)
    logging.info(f"[Results] Average MSE after compression: {average_mse_compressed:.5f}")

    retrieval_time = time.time() - start_time
    logging.info(f"[Performance] Total execution time: {retrieval_time:.2f} seconds.")

if __name__ == "__main__":
    test_scaled_holographic_memory()
