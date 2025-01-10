import sqlite3

def initialize_database(db_path):
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                input TEXT,
                output TEXT,
                domain TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    print(f"Database initialized at {db_path}.")

# Initialize databases
initialize_database("data/entity_memory.db")
initialize_database("data/meta_memory.db")
