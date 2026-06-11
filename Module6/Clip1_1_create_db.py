# Creating and populating a SQLite database with sample sales data

import sqlite3

conn = sqlite3.connect("sales.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    region TEXT,
    sales INTEGER
)
""")

cursor.execute("DELETE FROM orders")  # clear old data

cursor.executemany(
    "INSERT INTO orders (region, sales) VALUES (?, ?)",
    [
    ("West", 100),
    ("East", 150),
    ("South", 130),
    ("West", 200),
    ("North", 180),
    ("East", 300),
    ("South", 170),
    ("West", 150),
    ("North", 220),
    ("East", 250)
    ]
)

conn.commit()
conn.close()

print("Database created.")