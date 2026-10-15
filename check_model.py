
import pickle
from pathlib import Path

model_dir = (
    Path(__file__).parent
    / "wiki_full_results"
    / "model"
)

with open(model_dir / "unigrams.pkl", "rb") as f:
    unigrams = pickle.load(f)

words = [
    "台灣",
    "臺灣",
    "台湾",
    "臺",
    "灣",
    "中興",
    "中興大學",
    "大學",
    "的"
]

print("===== 不同詞彙的出現次數 =====")

for word in words:
    print(f"{word}: {unigrams[word]:,}")


import jieba

sentences = [
    "臺灣位於亞洲東部",
    "台灣是一個美麗的地方",
    "國立中興大學位於臺中市",
    "臺灣的歷史文化非常豐富"
]

print("\n===== jieba 斷詞測試 =====")

for sentence in sentences:
    print("原句：", sentence)
    print("斷詞：", list(jieba.cut(sentence)))
    print()
