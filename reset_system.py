# reset_system.py

import os
import sqlite3
import numpy as np

# List of database files to reset
DATABASE_FILES = [
    "data/meta_memory.db",
    "data/entity_memory.db",
    "data/math.db",
    "data/english.db",
    "data/programming.db",
    "data/science.db",
]

# List of holographic memory files to reset
HOLOGRAPHIC_MEMORY_FILES = [
    "data/holographic_memory.npy",
    "data/MetaEntity1_holographic_memory.npy",
    "data/MetaEntity2_holographic_memory.npy",
    "data/SuperEntity1_holographic_memory.npy",
    "data/SuperEntity2_holographic_memory.npy",
    "data/NormalEntity1_holographic_memory.npy",
    "data/NormalEntity2_holographic_memory.npy",
]

def reset_database(db_path):
    """
    Reset a database by dropping all tables and recreating the necessary structure.
    :param db_path: Path to the database file.
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop all tables (if they exist)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
            print(f"Dropped table: {table[0]} in {db_path}")

        # Recreate the necessary table structure
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                input TEXT,
                output TEXT,
                domain TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print(f"Recreated table 'knowledge' in {db_path}")

        # Commit changes and close the connection
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error resetting database {db_path}: {e}")

def reset_holographic_memory(file_path):
    """
    Reset a holographic memory file by overwriting it with an empty array.
    :param file_path: Path to the holographic memory file.
    """
    try:
        # Overwrite the file with an empty array
        np.save(file_path, np.zeros(16384, dtype=complex))
        print(f"Reset holographic memory: {file_path}")
    except Exception as e:
        print(f"Error resetting holographic memory {file_path}: {e}")

def reset_system():
    """
    Reset the system by dropping all database tables and clearing holographic memory.
    """
    print("Resetting the system...")

    # Reset databases
    print("\nResetting databases...")
    for db_file in DATABASE_FILES:
        reset_database(db_file)

    # Reset holographic memory
    print("\nResetting holographic memory...")
    for memory_file in HOLOGRAPHIC_MEMORY_FILES:
        reset_holographic_memory(memory_file)

    print("\nSystem reset complete. All data has been cleared.")

if __name__ == "__main__":
    reset_system()