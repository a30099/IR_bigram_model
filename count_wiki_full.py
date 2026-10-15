
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
    with conn:
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


def count_wiki_full(data_path, db_path, total_articles, batch_size):
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

    milestones = {
        int(total_articles * 0.01): "1%",
        int(total_articles * 0.10): "10%",
        total_articles: "100%"
    }

    results = []

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

            is_milestone = article_count in milestones

            if article_count % batch_size == 0 or is_milestone:
                save_counts(conn, unigrams, bigrams)

                unigrams.clear()
                bigrams.clear()

            if is_milestone:
                V = conn.execute(
                    "SELECT COUNT(*) FROM unigrams"
                ).fetchone()[0]

                B = conn.execute(
                    "SELECT COUNT(*) FROM bigrams"
                ).fetchone()[0]

                elapsed = time.perf_counter() - start_time
                percentage = milestones[article_count]

                results.append(
                    (percentage, article_count, N, V, B, elapsed)
                )

                print(f"\n===== {percentage} 語料統計 =====")
                print(f"文章數：{article_count:,}")
                print(f"N：{N:,}")
                print(f"V：{V:,}")
                print(f"B：{B:,}")
                print(f"時間：{elapsed:.2f} 秒\n", flush=True)

            elif article_count % 10000 == 0:
                print(f"已處理 {article_count:,} 篇文章", flush=True)

    with open(db_path.parent / "wiki_statistics.csv", "w", encoding="utf-8-sig") as f:
        f.write("percentage,articles,N,V,B,time_seconds\n")

        for percentage, articles, n, v, b, elapsed in results:
            f.write(
                f"{percentage},{articles},{n},{v},{b},{elapsed:.2f}\n"
            )

    conn.close()

    print("===== 完整語料統計完成 =====")
    print("資料庫：", db_path)
    print("統計表：", db_path.parent / "wiki_statistics.csv")


data_dir = Path(__file__).parent

count_wiki_full(
    data_path=data_dir / "wiki_article_list_2023_tra.json",
    db_path=data_dir / "wiki_counts_full.db",
    total_articles=1323055,
    batch_size=1000
)
