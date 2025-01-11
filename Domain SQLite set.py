import sqlite3

def initialize_database(db_path):
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                input TEXT,
                output TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    print(f"Database initialized at {db_path}.")

# Initialize domain-specific databases
initialize_database("data/math.db")
initialize_database("data/english.db")
initialize_database("data/programming.db")
initialize_database("data/science.db")