
import pickle
import zlib
from pathlib import Path
from heapq import nlargest


def predict(word, k, model_dir, bucket_count):
    # 讀取 Unigram
    with open(model_dir / "unigrams.pkl", "rb") as f:
        unigrams = pickle.load(f)

    # 找到該詞所在的 Bigram 分桶
    bucket_id = zlib.crc32(
        word.encode("utf-8")
    ) % bucket_count

    bucket_path = (
        model_dir / f"bigram_{bucket_id:03d}.pkl"
    )

    with open(bucket_path, "rb") as f:
        bigrams = pickle.load(f)

    # 查詢該詞出現次數
    word_count = unigrams[word]

    if word_count == 0:
        return []

    # 取得最常出現的後續詞
    candidates = (
        (next_word, count)
        for (first_word, next_word), count in bigrams.items()
        if first_word == word
    )

    top_k = nlargest(
        k,
        candidates,
        key=lambda item: item[1]
    )

    return [
        (next_word, count / word_count)
        for next_word, count in top_k
    ]


if __name__ == "__main__":
    model_dir = (
        Path(__file__).parent
        / "wiki_full_results"
        / "model"
    )

    results = predict(
        word="台灣",
        k=10,
        model_dir=model_dir,
        bucket_count=64
    )

    print("===== 下一個詞預測 =====")

    for word, probability in results:
        print(f"{word}: {probability:.6f}")
