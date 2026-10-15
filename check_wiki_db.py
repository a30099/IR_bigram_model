
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "wiki_counts_test.db"

conn = sqlite3.connect(db_path)

print("===== SQLite 統計結果 =====")

N = conn.execute(
    "SELECT SUM(count) FROM unigrams"
).fetchone()[0]

V = conn.execute(
    "SELECT COUNT(*) FROM unigrams"
).fetchone()[0]

B = conn.execute(
    "SELECT COUNT(*) FROM bigrams"
).fetchone()[0]

print(f"N（總詞數）：{N:,}")
print(f"V（相異詞數）：{V:,}")
print(f"B（相異 Bigram 數）：{B:,}")

print("\n===== 最常見的 10 個詞 =====")

rows = conn.execute("""
    SELECT word, count
    FROM unigrams
    ORDER BY count DESC
    LIMIT 10
""").fetchall()

for word, count in rows:
    print(f"{word}: {count:,}")

print("\n===== 最常見的 10 個 Bigram =====")

rows = conn.execute("""
    SELECT word1, word2, count
    FROM bigrams
    ORDER BY count DESC
    LIMIT 10
""").fetchall()

for word1, word2, count in rows:
    print(f"({word1}, {word2}): {count:,}")

conn.close()
