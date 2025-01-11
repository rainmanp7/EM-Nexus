-- math.db
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY,
    input TEXT,
    output TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);