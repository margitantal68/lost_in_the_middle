"""
Diagrams comparing SQuAD vs LiveRAG results:
1. Retrieval performance (MRR/Recall@k, Top-k vs reranked)
2. Lost-in-the-middle curves (score vs. position, per model)
3. Answer-quality bars (Top-k vs Reranked, per model)
"""
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------
# 1. Retrieval performance
# ---------------------------------------------------------------
metrics = ["MRR", "R@1", "R@2", "R@3", "R@4", "R@5"]
squad_topk    = [0.8649, 0.8957, 0.9263, 0.9263, 0.9464, 0.9558]
squad_rerank  = [0.9301, 0.9082, 0.9460, 0.9526, 0.9548, 0.9558]
liverag_topk   = [0.9291, 0.8971, 0.9420, 0.9565, 0.9723, 0.9763]
liverag_rerank = [0.9557, 0.9380, 0.9683, 0.9749, 0.9763, 0.9763]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
x = np.arange(len(metrics))
width = 0.35

for ax, topk, rerank, title in [
    (axes[0], squad_topk, squad_rerank, "SQuAD"),
    (axes[1], liverag_topk, liverag_rerank, "LiveRAG"),
]:
    ax.bar(x - width/2, topk, width, label="Top-$k$", color="#8fa8c9")
    ax.bar(x + width/2, rerank, width, label="Reranked", color="#2b5f8e")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title(title)
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Score")
axes[0].legend(loc="lower right")
fig.suptitle("Retrieval Performance: Top-$k$ vs. Reranked (bge-reranker-v2-m3)")
fig.tight_layout()
fig.savefig("plots/retrieval_comparison.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------
# 2. Lost-in-the-middle curves
# ---------------------------------------------------------------
positions = ["None", "First", "Middle", "Last", "Crossencoder"]

squad_scores = {
    "Llama":   [4.1803, 4.1699, 4.1240, 4.1660, 4.1768],
    "Gemma":   [4.4693, 4.4828, 4.4477, 4.4548, 4.4717],
    "Mistral": [4.2200, 4.2556, 4.1457, 4.2163, 4.2505],
}
liverag_scores = {
    "Llama":   [3.9800, 4.0277, 3.6544, 3.6847, 4.0100],
    "Mistral": [4.2401, 4.2836, 4.0989, 4.2586, 4.2757],
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
colors = {"Llama": "#d1495b", "Gemma": "#557153", "Mistral": "#3f88c5"}

for model, scores in squad_scores.items():
    axes[0].plot(positions, scores, marker="o", label=model, color=colors[model])
axes[0].set_title("SQuAD")
axes[0].set_ylabel("Judged score (1-5)")

for model, scores in liverag_scores.items():
    axes[1].plot(positions, scores, marker="o", label=model, color=colors[model])
axes[1].set_title("LiveRAG")

for ax in axes:
    ax.set_ylim(3.5, 4.6)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.axvspan(1.5, 2.5, color="grey", alpha=0.08)  # highlight "Middle"

fig.suptitle("Lost-in-the-Middle Effect: Answer Quality by Context Position")
fig.tight_layout()
fig.savefig("plots/litm_comparison.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------
# 3. Answer-quality bars (Top-k vs Reranked)
# ---------------------------------------------------------------
squad_models = ["Llama", "Mistral", "Gemma"]
squad_topk_q   = [4.1803, 4.2200, 4.4693]
squad_topk_std = [1.1904, 1.1300, 1.0655]
squad_rr_q     = [4.1768, 4.2505, 4.4717]
squad_rr_std   = [1.1993, 1.1157, 1.0675]

liverag_models = ["Llama", "Mistral"]
liverag_topk_q   = [3.9800, 4.2401]
liverag_topk_std = [1.0475, 0.7321]
liverag_rr_q     = [4.0100, 4.2757]
liverag_rr_std   = [1.0700, 0.6612]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

for ax, models, topk_q, topk_std, rr_q, rr_std, title in [
    (axes[0], squad_models, squad_topk_q, squad_topk_std, squad_rr_q, squad_rr_std, "SQuAD"),
    (axes[1], liverag_models, liverag_topk_q, liverag_topk_std, liverag_rr_q, liverag_rr_std, "LiveRAG"),
]:
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, topk_q, width, yerr=topk_std, capsize=4, label="Top-$k$", color="#8fa8c9")
    ax.bar(x + width/2, rr_q, width, yerr=rr_std, capsize=4, label="Reranked", color="#2b5f8e")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Judged score (1-5)")
axes[0].legend(loc="lower right")
fig.suptitle("Answer Quality: Top-$k$ vs. Reranked Context (error bars = std)")
fig.tight_layout()
fig.savefig("plots/answer_quality_comparison.png", dpi=200)
plt.close(fig)

print("Saved 3 figures to plots folder")