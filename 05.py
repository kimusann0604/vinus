import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

# データの準備
reasons = [
    "手術費用がかかる", "失敗するのが怖い", "健康なのに手術をしたくない",
    "手術が怖い", "痛みをともなう", "親からもらった身体へメスを入れる抵抗感",
    "手術後のダウンタイム", "いまの自分に自信がある", "周りの人の目が気になる", "その他"
]

male_data = [
    [50.0, 48.6, 43.2], [41.3, 22.9, 34.1], [8.7, 22.9, 22.7],
    [19.6, 22.9, 29.6], [13.0, 22.9, 11.4], [23.9, 22.9, 11.4],
    [13.0, 14.3, 13.6], [19.6, 20.0, 6.8], [19.6, 11.4, 9.1],
    [0.0, 5.7, 2.3]
]

female_data = [
    [53.3, 66.7, 45.5], [33.3, 59.3, 40.9], [40.0, 22.2, 25.0],
    [46.7, 40.7, 50.0], [43.3, 51.9, 43.2], [50.0, 37.0, 29.6],
    [20.0, 40.7, 20.5], [6.7, 14.8, 6.8], [30.0, 11.1, 4.6],
    [3.3, 3.7, 9.1]
]

# グラフの設定
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
x = np.arange(len(reasons))
width = 0.25

# 男性のグラフ
ax1.bar(x - width, [d[0] for d in male_data], width, label='10代', color='skyblue')
ax1.bar(x, [d[1] for d in male_data], width, label='20代', color='royalblue')
ax1.bar(x + width, [d[2] for d in male_data], width, label='30代', color='navy')

ax1.set_ylabel('回答率 (%)')
ax1.set_title('男性', loc='left')
ax1.set_xticks(x)
ax1.set_xticklabels(reasons, rotation=45, ha='right')
ax1.legend()

# 女性のグラフ
ax2.bar(x - width, [d[0] for d in female_data], width, label='10代', color='lightpink')
ax2.bar(x, [d[1] for d in female_data], width, label='20代', color='hotpink')
ax2.bar(x + width, [d[2] for d in female_data], width, label='30代', color='deeppink')

ax2.set_ylabel('回答率 (%)')
ax2.set_xlabel('理由')
ax2.set_title('女性', loc='left')
ax2.set_xticks(x)
ax2.set_xticklabels(reasons, rotation=45, ha='right')
ax2.legend()

plt.suptitle('「美容整形」をしたくない理由（複数回答）', fontsize=16)
plt.tight_layout()
plt.show()