import sqlite3
from contextlib import closing
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "chat_history.db"


def _connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def initialize_history(db_path: str | Path = DB_PATH) -> None:
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def save_exchange(
    question: str, answer: str, db_path: str | Path = DB_PATH
) -> None:
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute(
                "INSERT INTO chat_history (question, answer) VALUES (?, ?)",
                (question, answer),
            )


def get_history(
    limit: int = 50, db_path: str | Path = DB_PATH
) -> list[dict[str, int | str]]:
    with closing(_connect(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT id, question, answer, created_at
            FROM chat_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {"id": row[0], "question": row[1], "answer": row[2], "created_at": row[3]}
        for row in rows
    ]


def clear_history(db_path: str | Path = DB_PATH) -> None:
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("DELETE FROM chat_history")
