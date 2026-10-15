
from pathlib import Path

data_path = Path(__file__).parent / "wiki_article_list_2023_tra.json"

total_articles = 0

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line in ("[", "]"):
            continue

        total_articles += 1

print(f"維基百科總文章數：{total_articles:,}")
print(f"1% 文章數：{int(total_articles * 0.01):,}")
print(f"10% 文章數：{int(total_articles * 0.10):,}")
print(f"100% 文章數：{total_articles:,}")
