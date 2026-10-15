
import pickle
import random
import zlib
from pathlib import Path


def load_next_words(word, model_dir, bucket_count):
    bucket_id = (
        zlib.crc32(word.encode("utf-8"))
        % bucket_count
    )

    bucket_path = model_dir / f"bigram_{bucket_id:03d}.pkl"

    with open(bucket_path, "rb") as f:
        bigrams = pickle.load(f)

    return [
        (word2, count)
        for (word1, word2), count in bigrams.items()
        if word1 == word
    ]


def generate(seed, n, model_dir, bucket_count, method):
    words = [seed]
    current_word = seed

    for _ in range(n):
        candidates = load_next_words(
            current_word, model_dir, bucket_count
        )

        if not candidates:
            break

        if method == "greedy":
            next_word = max(
                candidates,
                key=lambda item: item[1]
            )[0]

        else:
            next_words = [word for word, count in candidates]
            weights = [count for word, count in candidates]

            next_word = random.choices(
                next_words,
                weights=weights,
                k=1
            )[0]

        if next_word == "</s>":
            break

        words.append(next_word)
        current_word = next_word

    return "".join(words)


if __name__ == "__main__":
    model_dir = (
        Path(__file__).parent
        / "wiki_full_results"
        / "model"
    )


    print("===== Greedy =====")
    print(generate("你", 30, model_dir, 64, "greedy"))

    print("\n===== Sampling =====")
    print(generate("你", 30, model_dir, 64, "sampling"))

    print("===== Greedy =====")
    print(generate("從", 30, model_dir, 64, "greedy"))

    print("\n===== Sampling =====")
    print(generate("從", 30, model_dir, 64, "sampling"))

    print("===== Greedy =====")
    print(generate("桃園", 30, model_dir, 64, "greedy"))

    print("\n===== Sampling =====")
    print(generate("桃園", 30, model_dir, 64, "sampling"))
    print("===== Greedy =====")
    print(generate("到", 30, model_dir, 64, "greedy"))

    print("\n===== Sampling =====")
    print(generate("到", 30, model_dir, 64, "sampling"))
    print("===== Greedy =====")
    print(generate("新竹", 30, model_dir, 64, "greedy"))

    print("\n===== Sampling =====")
    print(generate("新竹", 30, model_dir, 64, "sampling"))
