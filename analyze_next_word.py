
import math
import pickle
import zlib
from pathlib import Path


def analyze_word(word, model_dir, bucket_count, top_k):
    with open(model_dir / "unigrams.pkl", "rb") as f:
        unigrams = pickle.load(f)

    bucket_id = (
        zlib.crc32(word.encode("utf-8"))
        % bucket_count
    )

    with open(
        model_dir / f"bigram_{bucket_id:03d}.pkl", "rb"
    ) as f:
        bigrams = pickle.load(f)

    next_words = [
        (word2, count)
        for (word1, word2), count in bigrams.items()
        if word1 == word
    ]

    total = sum(count for _, count in next_words)

    entropy = -sum(
        (count / total) * math.log2(count / total)
        for _, count in next_words
    )

    top_words = sorted(
        next_words,
        key=lambda item: item[1],
        reverse=True
    )[:top_k]

    print(f"\n===== {word} =====")
    print(f"Unigram Count：{unigrams[word]:,}")
    print(f"Next-word Types：{len(next_words):,}")
    print(f"Entropy：{entropy:.4f} bits")

    print(f"\n最常見的 {top_k} 個後續詞：")

    for next_word, count in top_words:
        probability = count / total
        print(
            f"{next_word}: "
            f"count={count:,}, "
            f"P={probability:.6f}"
        )


if __name__ == "__main__":
    model_dir = (
        Path(__file__).parent
        / "wiki_full_results"
        / "model"
    )

    analyze_word(
        word="的",
        model_dir=model_dir,
        bucket_count=64,
        top_k=10
    )

    analyze_word(
        word="台灣",
        model_dir=model_dir,
        bucket_count=64,
        top_k=10
    )
