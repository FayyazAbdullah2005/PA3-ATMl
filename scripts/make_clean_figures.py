"""Generate publication-quality, perfectly legible vector figures for ATML PA3."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_FIGS = Path("report/figures")
MAIN_FIGS = Path("main/figures")
REPORT_FIGS.mkdir(parents=True, exist_ok=True)
MAIN_FIGS.mkdir(parents=True, exist_ok=True)

# Set global matplotlib parameters for publication graphics
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.5,
    "legend.title_fontsize": 8.0,
    "figure.titlesize": 10.0,
    "lines.linewidth": 1.8,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def save_fig(fig, base_name):
    for d in (REPORT_FIGS, MAIN_FIGS):
        fig.savefig(d / f"{base_name}.pdf", format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(d / f"{base_name}.png", format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {base_name}.pdf and {base_name}.png")

def ema(values, alpha=0.08):
    smoothed = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    return np.array(smoothed)

# -------------------------------------------------------------------------
# Figure 1: Task 1 SFT Loss & L2-SP Objective
# -------------------------------------------------------------------------
print("--> Generating Task 1 figures...")
methods = [
    ("full", "Full FT", "#0072B2", "-"),
    ("l2sp", "Full FT + L2-SP", "#E69F00", "-"),
    ("lora4", "LoRA (r=4)", "#009E73", "-"),
    ("lora16", "LoRA (r=16)", "#D55E00", "-"),
    ("oplora", "OPLoRA (r=16, k=128)", "#7570B3", "-"),
]

# Task 1 SFT Loss
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for key, label, color, ls in methods:
    df = pd.read_csv(f"main/task1/{key}_training.csv")
    x = df["step"].to_numpy()
    y = df["sft_loss"].to_numpy()
    y_smooth = ema(y, alpha=0.08)
    ax.plot(x, y, color=color, alpha=0.15, linewidth=0.7)
    ax.plot(x, y_smooth, label=label, color=color, linestyle=ls, linewidth=1.8)

ax.set_xlabel("Optimizer Update", fontweight="bold")
ax.set_ylabel("Response Cross-Entropy", fontweight="bold")
ax.set_ylim(0.12, 1.05)
ax.set_xlim(0, 1250)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(framealpha=0.9, loc="upper right", borderpad=0.3, handletextpad=0.4, labelspacing=0.25)
save_fig(fig, "task1_sft")

# Task 1 L2-SP Objective Decomposition
fig, ax = plt.subplots(figsize=(3.4, 2.4))
df_l2sp = pd.read_csv("main/task1/l2sp_training.csv")
x = df_l2sp["step"].to_numpy()
sft_loss = df_l2sp["sft_loss"].to_numpy()
penalty = df_l2sp["weighted_penalty"].to_numpy()
total = df_l2sp["total_loss"].to_numpy()

sft_smooth = ema(sft_loss, alpha=0.08)
total_smooth = ema(total, alpha=0.08)

ax.plot(x, sft_loss, color="#0072B2", alpha=0.12, linewidth=0.7)
ax.plot(x, sft_smooth, label=r"SFT Loss ($\mathcal{L}_{\mathrm{SFT}}$)", color="#0072B2", linewidth=1.8)
ax.plot(x, penalty, label=r"Penalty ($\lambda\Omega$)", color="#D55E00", linewidth=1.8)
ax.plot(x, total_smooth, label=r"Total Objective ($\mathcal{L} + \lambda\Omega$)", color="#009E73", linestyle="--", linewidth=1.8)

ax.set_xlabel("Optimizer Update", fontweight="bold")
ax.set_ylabel("Objective Loss", fontweight="bold")
ax.set_ylim(-0.02, 1.05)
ax.set_xlim(0, 1250)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(framealpha=0.9, loc="upper right", borderpad=0.3, handletextpad=0.4, labelspacing=0.25)
save_fig(fig, "task1_l2sp_objective")

# -------------------------------------------------------------------------
# Figure 2: Task 2 Multi-Stage Curriculum Loss & Development Metrics (196 Tokens)
# -------------------------------------------------------------------------
print("--> Generating Task 2 (196 tokens) figures...")
for tokens, suffix in [(196, "196"), (49, "49")]:
    # Training loss
    df_t2 = pd.read_csv(f"main/task2/{tokens}/training.csv")
    x = df_t2["step"].to_numpy()
    y = df_t2["sft_loss"].to_numpy()
    y_smooth = ema(y, alpha=0.04)
    
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(x, y, color="#0072B2", alpha=0.12, linewidth=0.6)
    ax.plot(x, y_smooth, label=f"VLM ({tokens} tokens)", color="#0072B2", linewidth=1.8)
    
    # Mark stage boundaries
    ax.axvline(2250, color="#666666", linestyle=":", linewidth=1.0)
    ax.axvline(3750, color="#666666", linestyle=":", linewidth=1.0)
    ax.text(1125, 0.95, "Stage 1\n(Caption)", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7.0, color="#444444")
    ax.text(3000, 0.95, "Stage 2\n(Inst.)", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7.0, color="#444444")
    ax.text(4125, 0.95, "Stage 3\n(VQA)", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=7.0, color="#444444")
    
    ax.set_xlabel("Optimizer Update", fontweight="bold")
    ax.set_ylabel("Example-Weighted Loss", fontweight="bold")
    ax.set_xlim(0, 4500)
    ax.grid(True, linestyle="--", alpha=0.4)
    save_fig(fig, f"task2_{suffix}_loss")

    # Development metrics
    df_dev = pd.read_csv(f"main/task2/{tokens}/development.csv")
    u = df_dev["update"].to_numpy()
    
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(u, df_dev["caption/correct"], label="Caption Loss", color="#0072B2", marker="o", markersize=3.5, linewidth=1.5)
    ax.plot(u, df_dev["grounding_dev/correct"], label="Grounding Correct NLL", color="#009E73", marker="s", markersize=3.5, linewidth=1.5)
    ax.plot(u, df_dev["grounding_dev/swap"], label="Grounding Swap NLL", color="#D55E00", marker="^", markersize=3.5, linewidth=1.5)
    ax.plot(u, df_dev["natural_dev/correct"], label="Natural Correct NLL", color="#7570B3", marker="D", markersize=3.5, linewidth=1.5)
    ax.plot(u, df_dev["grounding_dev/swap_penalty"], label=r"Grounding $\Delta$NLL", color="#E69F00", linestyle="--", linewidth=1.6)
    
    ax.axvline(2250, color="#666666", linestyle=":", linewidth=0.9)
    ax.axvline(3750, color="#666666", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Optimizer Update", fontweight="bold")
    ax.set_ylabel("NLL / Cross-Entropy", fontweight="bold")
    ax.set_xlim(-50, 4550)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(framealpha=0.9, loc="upper right", fontsize=6.8, borderpad=0.25, labelspacing=0.2)
    save_fig(fig, f"task2_{suffix}_development")

# -------------------------------------------------------------------------
# Figure 3: Task 3 Test-Time Scaling (MATH & MathVista)
# -------------------------------------------------------------------------
print("--> Generating Task 3 figures...")
df_t3 = pd.read_parquet("main/task3-offline/summary.parquet")
df_t3_eval = df_t3[df_t3["split"] == "evaluation"]

for ds in ("math", "mathvista"):
    part = df_t3_eval[df_t3_eval["dataset"] == ds]
    
    # 1. Accuracy vs N
    fig, ax = plt.subplots(figsize=(2.5, 2.1))
    methods_plot = [
        ("first", "FirstAns", "#888888", ":", "o"),
        ("sc", "Self-Consistency", "#0072B2", "-", "s"),
        ("certainty", "Self-Certainty", "#009E73", "-", "^"),
        ("certainty_rwc" if ds=="math" else "visconf_rwc", "Certainty+RWC" if ds=="math" else "VisConf+RWC", "#D55E00", "-", "*"),
        ("logp", "Mean Logp", "#CC79A7", "--", "v"),
    ]
    if ds == "mathvista":
        methods_plot.append(("visconf", "VisConf", "#E69F00", "-.", "D"))
    
    for m, lbl, clr, ls, mrk in methods_plot:
        sub = part[part["method"] == m].sort_values("N")
        if len(sub) > 0:
            ax.plot(sub["N"], sub["accuracy"] * 100, label=lbl, color=clr, linestyle=ls, marker=mrk, markersize=3.5, linewidth=1.5)
            
    ax.set_xlabel("Candidate Budget N", fontweight="bold", fontsize=8.5)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(framealpha=0.9, loc="lower right", fontsize=6.2, borderpad=0.2, labelspacing=0.15)
    save_fig(fig, f"task3_{ds}_accuracy_vs_N")

    # 2. Accuracy vs Tokens
    fig, ax = plt.subplots(figsize=(2.5, 2.1))
    for m, lbl, clr, ls, mrk in methods_plot:
        sub = part[part["method"] == m].sort_values("tokens")
        if len(sub) > 0:
            ax.plot(sub["tokens"], sub["accuracy"] * 100, label=lbl, color=clr, linestyle=ls, marker=mrk, markersize=3.5, linewidth=1.5)
    ax.set_xlabel("Generated Tokens", fontweight="bold", fontsize=8.5)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    save_fig(fig, f"task3_{ds}_accuracy_vs_tokens")

    # 3. Oracle vs N
    fig, ax = plt.subplots(figsize=(2.5, 2.1))
    sub_sc = part[part["method"] == "sc"].sort_values("N")
    ax.plot(sub_sc["N"], sub_sc["oracle"] * 100, label="Oracle Coverage", color="#D55E00", marker="D", markersize=4.0, linewidth=1.8)
    ax.plot(sub_sc["N"], sub_sc["accuracy"] * 100, label="SC Accuracy", color="#0072B2", marker="s", markersize=3.5, linewidth=1.5)
    ax.fill_between(sub_sc["N"], sub_sc["accuracy"] * 100, sub_sc["oracle"] * 100, color="#D55E00", alpha=0.15, label="Selection Gap")
    
    ax.set_xlabel("Candidate Budget N", fontweight="bold", fontsize=8.5)
    ax.set_ylabel("Coverage / Accuracy (%)", fontweight="bold", fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(framealpha=0.9, loc="lower right", fontsize=6.5, borderpad=0.25, labelspacing=0.2)
    save_fig(fig, f"task3_{ds}_oracle_vs_N")

# -------------------------------------------------------------------------
# Figure 4: Trajectories (32-token rolling mean)
# -------------------------------------------------------------------------
print("--> Generating trajectory figures...")
# MATH rollout trajectory
df_traj_m = pd.read_csv("main/trajectories/math_65e29ecfae_rollout24.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.3))
t = df_traj_m["token"].to_numpy()
logp_smooth = df_traj_m["logp"].rolling(32, min_periods=1).mean()
kl_smooth = df_traj_m["kl_u_p"].rolling(32, min_periods=1).mean()

ax.plot(t, logp_smooth, label=r"Sampled Log-Prob ($L_{i,t}$)", color="#0072B2", linewidth=1.7)
ax.plot(t, kl_smooth, label=r"Self-Certainty ($K_{i,t}$)", color="#D55E00", linewidth=1.7)
ax.set_xlabel("Generated Token Index", fontweight="bold")
ax.set_ylabel("Signal (32-Token Rolling Mean)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(framealpha=0.9, loc="center right", fontsize=7.2)
save_fig(fig, "math_65e29ecfae_rollout24")

# MathVista rollout trajectory
df_traj_v = pd.read_csv("main/trajectories/mathvista_3e7c535ec4_rollout2.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.3))
t = df_traj_v["token"].to_numpy()
conf_smooth = df_traj_v["calibrated_confidence"].rolling(32, min_periods=1).mean()
vis_smooth = df_traj_v["visual_engagement"].rolling(32, min_periods=1).mean()

ax.plot(t, conf_smooth, label=r"Calibrated Confidence ($c_{i,t}$)", color="#009E73", linewidth=1.7)
ax.plot(t, vis_smooth, label=r"Visual Engagement ($v_{i,t}$)", color="#7570B3", linewidth=1.7)
ax.set_xlabel("Generated Token Index", fontweight="bold")
ax.set_ylabel("Signal (32-Token Rolling Mean)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(framealpha=0.9, loc="center right", fontsize=7.2)
save_fig(fig, "mathvista_3e7c535ec4_rollout2")

print("All figures successfully regenerated and saved to report/figures/ and main/figures/!")
