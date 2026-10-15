
import csv
import json
import pickle
import re
import struct
import time
import zlib
import ctypes
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import jieba


def init_worker():
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x00004000
    )
    jieba.initialize()


def read_articles(data_path):
    with open(data_path, encoding="utf-8") as f:
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


def make_batches(data_path, batch_size, milestones):
    articles = read_articles(data_path)
    start = 0

    for stage, end in enumerate(milestones):
        remaining = end - start

        while remaining:
            batch = list(islice(
                articles, min(batch_size, remaining)
            ))

            yield stage, batch
            remaining -= len(batch)

        start = end


def count_batch(task):
    stage, articles = task

    unigrams = Counter()
    bigrams = Counter()
    N = 0

    for article in articles:
        for tokens in tokenize_article(article):
            unigrams.update(tokens)
            bigrams.update(zip(tokens, tokens[1:]))
            N += len(tokens)

    return stage, len(articles), N, unigrams, bigrams


def bucket_id(word, bucket_count):
    return zlib.crc32(word.encode("utf-8")) % bucket_count


def write_chunk(path, data):
    payload = pickle.dumps(data, protocol=5)

    with open(path, "ab") as f:
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)


def read_chunks(path):
    if not path.exists():
        return

    with open(path, "rb") as f:
        while header := f.read(8):
            size = struct.unpack("<Q", header)[0]
            yield pickle.loads(f.read(size))


def flush_bigrams(bigrams, stage, temp_dir, bucket_count):
    buckets = [dict() for _ in range(bucket_count)]

    for (word1, word2), count in bigrams.items():
        i = bucket_id(word1, bucket_count)
        buckets[i][(word1, word2)] = count

    for i, data in enumerate(buckets):
        if data:
            path = temp_dir / f"stage{stage}_bucket{i:03d}.bin"
            write_chunk(path, data)

    bigrams.clear()


def run_full_count(
    data_path,
    output_dir,
    milestones,
    workers,
    batch_size,
    bucket_count,
    flush_limit
):
    start_time = time.perf_counter()

    output_dir.mkdir(parents=True, exist_ok=False)
    temp_dir = output_dir / "temp"
    model_dir = output_dir / "model"

    temp_dir.mkdir()
    model_dir.mkdir()

    all_unigrams = Counter()
    pending_bigrams = Counter()

    total_N = 0
    total_articles = 0
    current_stage = 0

    stage_N = [0] * len(milestones)
    stage_V = [0] * len(milestones)
    stage_B = [0] * len(milestones)

    batches = make_batches(data_path, batch_size, milestones)

    print("開始平行斷詞與統計...", flush=True)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker
    ) as executor:

        results = executor.map(
            count_batch,
            batches,
            buffersize=2
        )

        for stage, articles, N, unigrams, bigrams in results:
            if stage != current_stage:
                flush_bigrams(
                    pending_bigrams,
                    current_stage,
                    temp_dir,
                    bucket_count
                )
                current_stage = stage

            all_unigrams.update(unigrams)
            pending_bigrams.update(bigrams)

            total_N += N
            total_articles += articles

            if len(pending_bigrams) >= flush_limit:
                flush_bigrams(
                    pending_bigrams,
                    stage,
                    temp_dir,
                    bucket_count
                )

            if total_articles in milestones:
                flush_bigrams(
                    pending_bigrams,
                    stage,
                    temp_dir,
                    bucket_count
                )

                stage_N[stage] = total_N
                stage_V[stage] = len(all_unigrams)

                print(
                    f"完成 {total_articles:,} 篇文章",
                    flush=True
                )

            elif total_articles % 10000 < articles:
                print(
                    f"已處理 {total_articles:,} 篇文章",
                    flush=True
                )

    with open(model_dir / "unigrams.pkl", "wb") as f:
        pickle.dump(all_unigrams, f, protocol=5)

    print("斷詞完成，開始合併 Bigram...", flush=True)

    for i in range(bucket_count):
        counts = Counter()

        for stage in range(len(milestones)):
            path = temp_dir / f"stage{stage}_bucket{i:03d}.bin"

            for chunk in read_chunks(path):
                counts.update(chunk)

            stage_B[stage] += len(counts)

        with open(model_dir / f"bigram_{i:03d}.pkl", "wb") as f:
            pickle.dump(counts, f, protocol=5)

        print(
            f"已合併 {i + 1}/{bucket_count} 個分桶",
            flush=True
        )

    elapsed = time.perf_counter() - start_time

    with open(
        output_dir / "wiki_statistics.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["percentage", "articles", "N", "V", "B"])

        for stage, percentage in enumerate(["1%", "10%", "100%"]):
            writer.writerow([
                percentage,
                milestones[stage],
                stage_N[stage],
                stage_V[stage],
                stage_B[stage]
            ])

            print(
                f"{percentage}: "
                f"N={stage_N[stage]:,}, "
                f"V={stage_V[stage]:,}, "
                f"B={stage_B[stage]:,}"
            )

    with open(output_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "N": total_N,
                "V": len(all_unigrams),
                "B": stage_B[-1],
                "t_seconds": elapsed,
                "bucket_count": bucket_count,
                "articles": total_articles,
                "workers": workers,
                "tokenizer": "jieba"
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\n全部完成，總時間：{elapsed:.2f} 秒")
    print(f"結果儲存於：{output_dir}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent

    run_full_count(
        data_path=data_dir / "wiki_article_list_2023_tra.json",
        output_dir=data_dir / "wiki_full_results",
        milestones=[13230, 132305, 1323055],
        workers=2,
        batch_size=200,
        bucket_count=64,
        flush_limit=150000
    )
