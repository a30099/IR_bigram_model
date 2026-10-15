
import json
import re
import time
import jieba
from pathlib import Path
from collections import Counter
import pickle



def tokenize_article(article):
    # 根據句尾標點切句，保留連續標點
    sentences = re.findall(r"[^。！？!?]+[。！？!?]*|[。！？!?]+", article)

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        words = [
            word
            for word in jieba.cut(sentence)
            if word.strip()
        ]

        tokens = ["<s>"] + words + ["</s>"]

        yield tokens


def count_ngrams(data_path, max_articles):
    unigram_counts = Counter()
    bigram_counts = Counter()
    total_tokens = 0
    article_count = 0

    start_time = time.perf_counter()

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")

            if line in ("[", "]"):
                continue

            article = json.loads(line)

            for tokens in tokenize_article(article):
                unigram_counts.update(tokens)
                bigram_counts.update(zip(tokens, tokens[1:]))
                total_tokens += len(tokens)

            article_count += 1

            if article_count >= max_articles:
                break

    elapsed = time.perf_counter() - start_time

    return unigram_counts, bigram_counts, total_tokens, elapsed


data_path = Path(__file__).parent / "wiki_article_list_2023_tra.json"

unigrams, bigrams, N, t = count_ngrams(data_path, max_articles=10000)

V = len(unigrams)
B = len(bigrams)

print("===== Part 1 統計結果 =====")
print(f"N（總詞數）：{N:,}")
print(f"V（相異詞數）：{V:,}")
print(f"B（相異 Bigram 數）：{B:,}")
print(f"t（執行時間）：{t:.2f} 秒")

print("\n===== 最常見的 10 個詞 =====")
for word, count in unigrams.most_common(10):
    print(f"{word}: {count:,}")

print("\n===== 最常見的 10 個 Bigram =====")
for pair, count in bigrams.most_common(10):
    print(f"{pair}: {count:,}")

model_path = Path(__file__).parent / "bigram_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump({
        "unigrams": unigrams,
        "bigrams": bigrams,
        "N": N,
        "V": V,
        "B": B,
        "t": t
    }, f)

print(f"\n模型已儲存至：{model_path}")


# ===== Step 6：驗證統計邏輯 =====

test_sentences = [
    ["<s>", "我", "喜歡", "貓", "</s>"],
    ["<s>", "我", "喜歡", "狗", "</s>"],
    ["<s>", "我", "討厭", "下雨", "</s>"]
]

test_unigrams = Counter()
test_bigrams = Counter()

for tokens in test_sentences:
    test_unigrams.update(tokens)
    test_bigrams.update(zip(tokens, tokens[1:]))

print("\n===== 測試語料驗證 =====")
print("count(我) =", test_unigrams["我"])
print("count(我, 喜歡) =", test_bigrams[("我", "喜歡")])
print("count(我, 討厭) =", test_bigrams[("我", "討厭")])

print("P(喜歡 | 我) =",
      test_bigrams[("我", "喜歡")] / test_unigrams["我"])

print("P(討厭 | 我) =",
      test_bigrams[("我", "討厭")] / test_unigrams["我"])
