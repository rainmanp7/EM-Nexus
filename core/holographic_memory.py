# core/holographic_memory.py

import numpy as np
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HolographicMemory:
    def __init__(self, dimensions=16384, initial_regularization=1e-6, memory_file="data/holographic_memory.npy"):
        """
        Initialize holographic memory with a high-dimensional space.
        :param dimensions: Number of dimensions for memory representation.
        :param initial_regularization: Starting regularization value for iterative encoding.
        :param memory_file: File path to save/load the memory space for persistence.
        """
        self.dimensions = dimensions
        self.initial_regularization = initial_regularization
        self.memory_file = memory_file

        # Load memory space from disk if it exists, otherwise initialize to zero
        if os.path.exists(self.memory_file):
            logging.info(f"Loading holographic memory from {self.memory_file}...")
            self.memory_space = np.load(self.memory_file)
        else:
            logging.info(f"Initializing new holographic memory with {dimensions} dimensions.")
            self.memory_space = np.zeros(dimensions, dtype=complex)

    def save_memory(self):
        """
        Save the memory space to disk for persistence.
        """
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        np.save(self.memory_file, self.memory_space)
        logging.info(f"Holographic memory saved to {self.memory_file}.")

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
        self.save_memory()  # Save memory after encoding

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
        self.save_memory()  # Save memory after compression
        return compression_ratio