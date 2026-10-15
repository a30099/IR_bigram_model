
import math
import pickle
import zlib
from pathlib import Path

import jieba


def prob(sentence, model_dir, bucket_count):
    words = [
        word for word in jieba.cut(sentence)
        if word.strip()
    ]

    tokens = ["<s>"] + words + ["</s>"]

    with open(model_dir / "unigrams.pkl", "rb") as f:
        unigrams = pickle.load(f)

    log_probability = 0.0
    has_zero = False

    print(f"\n句子：{sentence}")
    print(f"斷詞結果：{tokens}")
    print("\nBigram 統計：")

    # 一次只保留一個分桶，避免記憶體持續增加
    current_bucket_id = None
    bigrams = None

    for word1, word2 in zip(tokens, tokens[1:]):
        bucket_id = (
            zlib.crc32(word1.encode("utf-8"))
            % bucket_count
        )

        if bucket_id != current_bucket_id:
            bucket_path = (
                model_dir / f"bigram_{bucket_id:03d}.pkl"
            )

            with open(bucket_path, "rb") as f:
                bigrams = pickle.load(f)

            current_bucket_id = bucket_id

        bigram_count = bigrams.get((word1, word2), 0)
        unigram_count = unigrams[word1]

        probability = (
            bigram_count / unigram_count
            if unigram_count > 0 else 0.0
        )

        print(
            f"({word1}, {word2}) "
            f"count={bigram_count:,} "
            f"P={probability:.8f}"
        )

        if probability == 0:
            has_zero = True
        else:
            log_probability += math.log(probability)

    if has_zero:
        log_probability = float("-inf")

    print(f"\nLog Probability：{log_probability:.6f}")

    return log_probability


if __name__ == "__main__":
    model_dir = (
        Path(__file__).parent
        / "wiki_full_results"
        / "model"
    )

    sentences = [
        "范耀中很厲害",
        "我喜歡資訊檢索這堂課",
        "臺灣是一個美麗的地方",
        "今天天氣很好",
        "今天的天氣是大學的學生"
    ]

    for sentence in sentences:
        prob(
            sentence=sentence,
            model_dir=model_dir,
            bucket_count=64
        )

        print("=" * 60)
