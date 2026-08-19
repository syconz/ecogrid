"""
db.py
SQLite persistence layer for EcoGrid: user accounts + per-user prediction
history. Uses Python's built-in sqlite3 (no new dependency needed) and
Werkzeug's password hashing (already installed as a Flask dependency).
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "ecogrid.db")


def get_db_connection():
    """Opens a new connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Creates tables if they don't exist yet. Safe to call on every app start."""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            building_type TEXT NOT NULL,
            square_feet REAL NOT NULL,
            zip_code TEXT,
            country TEXT,
            predicted_usage_kwh REAL NOT NULL,
            estimated_cost REAL NOT NULL,
            estimated_savings REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# User accounts
# ---------------------------------------------------------------------------

def create_user(username, password):
    """
    Creates a new user with a hashed password.
    Returns (True, user_id) on success, or (False, error_message) on failure
    (e.g. username already taken).
    """
    conn = get_db_connection()
    try:
        password_hash = generate_password_hash(password)
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return True, cursor.lastrowid
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    finally:
        conn.close()


def verify_user(username, password):
    """
    Checks username/password against the database.
    Returns the user's id (int) if valid, or None if invalid.
    """
    conn = get_db_connection()
    row = conn.execute(
        "SELECT id, password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if row is None:
        return None
    if not check_password_hash(row["password_hash"], password):
        return None
    return row["id"]


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------

def save_prediction(user_id, building_type, square_feet, zip_code, country,
                     predicted_usage_kwh, estimated_cost, estimated_savings):
    """Stores one prediction result tied to a user."""
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO predictions (
            user_id, building_type, square_feet, zip_code, country,
            predicted_usage_kwh, estimated_cost, estimated_savings
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, building_type, square_feet, zip_code, country,
          predicted_usage_kwh, estimated_cost, estimated_savings))
    conn.commit()
    conn.close()


def get_predictions_for_user(user_id, limit=30):
    """
    Returns the user's most recent predictions (most recent first from the
    query, but reversed before returning so the list is oldest-first — handy
    for charting in chronological order).
    """
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT building_type, square_feet, predicted_usage_kwh,
               estimated_cost, estimated_savings, created_at
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit)).fetchall()
    conn.close()

    history = [dict(row) for row in rows]
    history.reverse()  # oldest first
    return history