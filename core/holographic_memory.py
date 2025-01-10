import numpy as np
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
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
        self.memory_space = np.zeros(dimensions, dtype=complex)  # Memory space
        self.initial_regularization = initial_regularization

    def normalize(self, vector):
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector

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

    def dynamic_encode(self, key, value, max_iterations=10, tolerance=1e-4):
        """
        Dynamically encode a key-value pair using adaptive regularization.
        :param key: Key vector (1D array).
        :param value: Value vector (1D array).
        :param max_iterations: Maximum number of encoding iterations.
        :param tolerance: Tolerance for convergence.
        """
        for i in range(max_iterations):
            regularization = self.initial_regularization * np.exp(-i / 10)  # Adaptive regularization
            previous_memory = self.memory_space.copy()
            self.encode(key, value, regularization)
            if np.linalg.norm(self.memory_space - previous_memory) < tolerance:
                logging.info(f"[HolographicMemory] Converged after {i + 1} iterations.")
                break

    def compress_memory(self, threshold=None):
        """
        Compress the memory by removing low-magnitude elements.
        :param threshold: Threshold for compression. If None, use adaptive thresholding.
        :return: Compression ratio (percentage of memory retained).
        """
        if threshold is None:
            threshold = np.percentile(np.abs(self.memory_space), 75)  # Retain 25% of memory
        mask = np.abs(self.memory_space) > threshold
        retained_elements = np.sum(mask)
        compression_ratio = retained_elements / self.dimensions
        self.memory_space[~mask] = 0
        return compression_ratio

# Example Usage
if __name__ == "__main__":
    # Initialize holographic memory
    memory = HolographicMemory(dimensions=16384)

    # Generate random key and value
    key = np.random.randn(16384)
    value = np.random.randn(16384)

    # Encode the key-value pair
    logging.info("Encoding key-value pair...")
    start_time = time.time()
    memory.dynamic_encode(key, value)
    encoding_time = time.time() - start_time
    logging.info(f"Encoding completed in {encoding_time:.4f} seconds.")

    # Retrieve the value
    logging.info("Retrieving value...")
    start_time = time.time()
    retrieved_value = memory.retrieve(key)
    retrieval_time = time.time() - start_time
    logging.info(f"Retrieval completed in {retrieval_time:.4f} seconds.")

    # Calculate MSE between original and retrieved value
    mse = np.mean((value - retrieved_value) ** 2)
    logging.info(f"Mean Squared Error (MSE): {mse:.4f}")

    # Print retrieved value (first 5 elements)
    print(f"Retrieved value (first 5 elements): {retrieved_value[:5]}")

    # Compress memory and measure compression ratio
    logging.info("Compressing memory...")
    start_time = time.time()
    compression_ratio = memory.compress_memory()
    compression_time = time.time() - start_time
    logging.info(f"Compression completed in {compression_time:.4f} seconds.")
    logging.info(f"Compression ratio: {compression_ratio * 100:.2f}% of memory retained.")

    # Retrieve after compression
    start_time = time.time()
    retrieved_value_compressed = memory.retrieve(key)
    retrieval_time_compressed = time.time() - start_time
    logging.info(f"Retrieval after compression completed in {retrieval_time_compressed:.4f} seconds.")

    # Calculate MSE after compression
    mse_compressed = np.mean((value - retrieved_value_compressed) ** 2)
    logging.info(f"Mean Squared Error (MSE) after compression: {mse_compressed:.4f}")

    # Print retrieved value after compression (first 5 elements)
    print(f"Retrieved value after compression (first 5 elements): {retrieved_value_compressed[:5]}")