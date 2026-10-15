
import matplotlib.pyplot as plt
from pathlib import Path

# 你的實際實驗數據
percentages = [1, 10, 100]
bigram_counts = [4370343, 12507653, 60836882]

# 換算為百萬，方便閱讀
bigram_millions = [count / 1_000_000 for count in bigram_counts]

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    percentages,
    bigram_millions,
    marker="o",
    markersize=9,
    linewidth=2.5,
    color="#2878B5"
)

# 標示每個實驗數值
for x, y in zip(percentages, bigram_millions):
    ax.annotate(
        f"{y:.2f}M",
        (x, y),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=11
    )

ax.set_title(
    "Growth of Unique Word Bigrams",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Corpus Size (% of Articles)", fontsize=12)
ax.set_ylabel("Unique Bigrams (Millions)", fontsize=12)

ax.set_xticks(percentages)
ax.set_xticklabels(["1%", "10%", "100%"])

ax.set_xlim(0, 105)
ax.set_ylim(0, 70)

ax.grid(True, linestyle="--", alpha=0.3)

fig.tight_layout()

# 儲存高解析度圖片
output_path = Path(__file__).parent / "bigram_growth.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"圖片已儲存至：{output_path}")
