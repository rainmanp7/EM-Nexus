# memory_store.py

import os
import sqlite3
import numpy as np
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
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


class MemoryStore:
    def __init__(self, db_path, holographic_dimensions=16384, regularisation=0.01):
        self.db_path = db_path
        self.ensure_directory_exists()
        self.conn = sqlite3.connect(db_path)
        self.holographic_memory = HolographicMemory(dimensions=holographic_dimensions, initial_regularization=regularisation)
        self._initialize_db()

    def ensure_directory_exists(self):
        """Ensure the directory for the database exists."""
        directory = os.path.dirname(self.db_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _initialize_db(self):
        """Initialize the SQLite database with the required table."""
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge (
                        id INTEGER PRIMARY KEY,
                        input TEXT,
                        output TEXT,
                        domain TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            logging.info(f"Database initialized at {self.db_path}.")
        except sqlite3.Error as e:
            logging.error(f"Database initialization failed: {e}")

    def store_knowledge(self, input_data, output_data, domain):
        try:
            with self.conn:
                self.conn.execute("""
                    INSERT INTO knowledge (input, output, domain) VALUES (?, ?, ?)
                """, (input_data, output_data, domain))
            logging.info(f"Knowledge stored: {input_data} -> {output_data} in domain {domain}")
        except sqlite3.Error as e:
            logging.error(f"Failed to store knowledge: {e}")

        # Holographic storage
        key = self._text_to_vector(input_data)
        value = self._text_to_vector(output_data)
        self.holographic_memory.dynamic_encode(key, value)

    def retrieve_holographic(self, query_text):
        """
        Retrieve knowledge using holographic memory.
        :param query_text: Text query to find matching knowledge.
        :return: Retrieved knowledge as text.
        """
        try:
            query_vector = self._text_to_vector(query_text)
            result_vector = self.holographic_memory.retrieve(query_vector)
            return self._vector_to_text(result_vector)
        except Exception as e:
            logging.error(f"Failed to retrieve knowledge: {e}")
            return None

    def close(self):
        """
        Close the database connection and perform any necessary cleanup.
        """
        if self.conn:
            self.conn.close()
            logging.info(f"Database connection closed for {self.db_path}.")

    @staticmethod
    def _text_to_vector(text, dimensions=1024):
        """
        Convert text into a high-dimensional vector.
        """
        if isinstance(text, np.ndarray):
            # Convert numpy array to string
            text = str(text.tolist())  # Convert array to list, then to string
        np.random.seed(hash(text) % (2**32))  # Consistent hash for text
        return np.random.randn(dimensions)

    @staticmethod
    def _vector_to_text(vector):
        """
        Convert a vector back to a textual representation.
        """
        return f"Vector[{len(vector)} dimensions]: {np.round(vector[:5], 3)}..."


def initialize_database(db_path):
    """
    Initialize the database with the advanced MemoryStore.
    :param db_path: Path to the SQLite database.
    """
    store = MemoryStore(db_path, holographic_dimensions=16384, regularisation=0.01)
    store.close()  # Close the database connection after initialization