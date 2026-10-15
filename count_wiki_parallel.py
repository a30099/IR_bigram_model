
import json
import re
import time
import jieba
from pathlib import Path
from itertools import islice
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os

def read_articles(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")

            if line in ("[", "]"):
                continue

            yield json.loads(line)


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
            word for word in jieba.cut(sentence)
            if word.strip()
        ]

        yield ["<s>"] + words + ["</s>"]


def init_worker():
    import ctypes

    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        BELOW_NORMAL_PRIORITY_CLASS
    )

    jieba.initialize()

def count_batch(articles):
    unigrams = Counter()
    bigrams = Counter()
    N = 0

    for article in articles:
        for tokens in tokenize_article(article):
            unigrams.update(tokens)
            bigrams.update(zip(tokens, tokens[1:]))
            N += len(tokens)

    return unigrams, bigrams, N


def make_batches(articles, batch_size):
    while batch := list(islice(articles, batch_size)):
        yield batch


def count_parallel(data_path, max_articles, workers, batch_size):
    start_time = time.perf_counter()

    all_unigrams = Counter()
    all_bigrams = Counter()
    N = 0
    article_count = 0

    articles = islice(read_articles(data_path), max_articles)
    batches = make_batches(articles, batch_size)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker
    ) as executor:

        results = executor.map(
            count_batch,
            batches,
            buffersize=8
        )

        for unigrams, bigrams, batch_N in results:
            all_unigrams.update(unigrams)
            all_bigrams.update(bigrams)
            N += batch_N

            article_count += unigrams["<s>"] * 0 + sum(
                1 for _ in ()
            ) if False else 0

    elapsed = time.perf_counter() - start_time

    return all_unigrams, all_bigrams, N, elapsed


if __name__ == "__main__":
    data_path = (
        Path(__file__).parent
        / "wiki_article_list_2023_tra.json"
    )

    unigrams, bigrams, N, t = count_parallel(
        data_path=data_path,
        max_articles=10000,
        workers=1,
        batch_size=200
    )

    print("\n===== Multi-processing 統計結果 =====")
    print(f"N：{N:,}")
    print(f"V：{len(unigrams):,}")
    print(f"B：{len(bigrams):,}")
    print(f"t：{t:.2f} 秒")
