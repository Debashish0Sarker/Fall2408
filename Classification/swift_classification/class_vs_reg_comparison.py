import matplotlib.pyplot as plt
import numpy as np

# =========================
# Top-k accuracies from: "thesis report changes and addition.pdf"
# =========================

models = [
    "CatBoostReg", "XGBoostReg", "RFReg",
    "CatBoostCls", "LogReg", "XGBCls"
]

top1 = np.array([47.17, 48.16, 41.97, 19.56, 19.56, 30.66], dtype=float)
top3 = np.array([87.17, 79.66, 100.0, 48.44, 46.67, 72.22], dtype=float)

x = np.arange(len(models))
width = 0.38

fig, ax = plt.subplots(figsize=(14, 6))

b1 = ax.bar(x - width/2, top1, width, label="Top-1 accuracy (%)")
b3 = ax.bar(x + width/2, top3, width, label="Top-3 accuracy (%)")

# Divider between regression (first 3) and classification (last 3)
ax.axvline(2.5, linestyle="--", linewidth=1)

# Group labels
ax.text(1.0, 103, "Regression Models", ha="center", va="bottom", fontsize=12)
ax.text(4.0, 103, "Classification Models", ha="center", va="bottom", fontsize=12)

ax.set_title("Top-1 vs Top-3 Accuracy: Regression vs Classification")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
ax.legend()

# Value labels on bars
for bars in (b1, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.2f}%",
                ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

# Optional save (high quality for thesis)
fig.savefig("topk_accuracy_combined_reg_vs_cls.png", dpi=300, bbox_inches="tight")
