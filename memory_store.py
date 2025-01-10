import os
import sqlite3
import networkx as nx
import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import qr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from sklearn.metrics import mean_squared_error
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


# Initialize databases
initialize_database("data/entity_memory.db")
initialize_database("data/meta_memory.db")


if __name__ == "__main__":
    # Test the holographic memory with a scaled test
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

    test_scaled_holographic_memory()