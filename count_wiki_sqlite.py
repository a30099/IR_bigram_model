
import json
import re
import time
import sqlite3
import jieba
from pathlib import Path
from collections import Counter


def tokenize_article(article):
    sentences = re.findall(
        r"[^。！？!?]+[。！？!?]*|[。！？!?]+",
        article
    )

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        words = [
            word
            for word in jieba.cut(sentence)
            if word.strip()
        ]

        yield ["<s>"] + words + ["</s>"]


def save_counts(conn, unigrams, bigrams):
    conn.executemany(
        """
        INSERT INTO unigrams (word, count)
        VALUES (?, ?)
        ON CONFLICT(word)
        DO UPDATE SET count = count + excluded.count
        """,
        unigrams.items()
    )

    conn.executemany(
        """
        INSERT INTO bigrams (word1, word2, count)
        VALUES (?, ?, ?)
        ON CONFLICT(word1, word2)
        DO UPDATE SET count = count + excluded.count
        """,
        (
            (word1, word2, count)
            for (word1, word2), count in bigrams.items()
        )
    )

    conn.commit()


def count_wiki(data_path, db_path, max_articles, batch_size):
    start_time = time.perf_counter()

    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE unigrams (
            word TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE bigrams (
            word1 TEXT,
            word2 TEXT,
            count INTEGER NOT NULL,
            PRIMARY KEY (word1, word2)
        )
    """)

    unigrams = Counter()
    bigrams = Counter()

    N = 0
    article_count = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")

            if line in ("[", "]"):
                continue

            article = json.loads(line)

            for tokens in tokenize_article(article):
                unigrams.update(tokens)
                bigrams.update(zip(tokens, tokens[1:]))
                N += len(tokens)

            article_count += 1

            if article_count % batch_size == 0:
                save_counts(conn, unigrams, bigrams)

                unigrams.clear()
                bigrams.clear()

                print(f"已處理 {article_count:,} 篇文章")

            if max_articles is not None and article_count >= max_articles:
                break

    if unigrams or bigrams:
        save_counts(conn, unigrams, bigrams)

    V = conn.execute(
        "SELECT COUNT(*) FROM unigrams"
    ).fetchone()[0]

    B = conn.execute(
        "SELECT COUNT(*) FROM bigrams"
    ).fetchone()[0]

    elapsed = time.perf_counter() - start_time

    conn.close()

    return N, V, B, elapsed


data_dir = Path(__file__).parent
data_path = data_dir / "wiki_article_list_2023_tra.json"
db_path = data_dir / "wiki_counts_test.db"

N, V, B, t = count_wiki(
    data_path=data_path,
    db_path=db_path,
    max_articles=1000,
    batch_size=100
)

print("\n===== Part 1 統計結果 =====")
print(f"N（總詞數）：{N:,}")
print(f"V（相異詞數）：{V:,}")
print(f"B（相異 Bigram 數）：{B:,}")
print(f"t（執行時間）：{t:.2f} 秒")
