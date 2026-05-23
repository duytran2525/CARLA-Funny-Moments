from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "imgs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Style
BG = "#0D1B2A"  # dark navy
CARD = "#1A2B3C"
BLUE = "#2E86DE"
CYAN = "#4ECDC4"
ORG = "#F7B731"
RED = "#EE5A24"
GRN = "#26de81"
WHT = "#E8EEF4"
GRY = "#6B7F93"
LBLUE = "#74B9FF"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    }
)


def fig(w=10, h=5):
    f, ax = plt.subplots(figsize=(w, h))
    f.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    return f, ax


def save(name):
    out = OUTPUT_DIR / f"{name}.png"
    plt.savefig(
        out,
        dpi=150,
        bbox_inches="tight",
        facecolor=BG,
        edgecolor="none",
    )
    plt.close()
    print(f"  generated {out}")


def draw_vehicle(ax, x, y, color, scale=1.0, zorder=5, alpha=1.0):
    """Small top-down car glyph drawn with patches to avoid emoji font issues."""
    w, h = 0.34 * scale, 0.54 * scale
    body = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={0.08 * scale}",
        fc=color,
        ec=WHT,
        lw=0.8 * scale,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(body)
    cabin = Rectangle(
        (x - w * 0.28, y - h * 0.1),
        w * 0.56,
        h * 0.25,
        fc=BG,
        ec="none",
        alpha=0.45 * alpha,
        zorder=zorder + 1,
    )
    ax.add_patch(cabin)
    nose = Circle((x, y + h * 0.33), 0.035 * scale, fc=WHT, ec="none", alpha=alpha, zorder=zorder + 1)
    ax.add_patch(nose)


# FIG 1 - Bird's Eye View
f, ax = fig(10, 6)
ax.set_xlim(-5, 5)
ax.set_ylim(-4, 4)

road_kw = dict(color="#1E3040", zorder=0)
ax.fill_betweenx([-4, 4], -1.5, 1.5, **road_kw)
ax.fill_between([-5, 5], -1.5, 1.5, **road_kw)
for y in [-0.05, 0.05]:
    ax.axhline(y, color="#2A4A60", lw=0.5, ls="--", zorder=1)
for x in [-0.05, 0.05]:
    ax.axvline(x, color="#2A4A60", lw=0.5, ls="--", zorder=1)

agents = {
    "Ego": (0, 0, ORG, 0.45),
    "A1": (2, 1.5, BLUE, 0.32),
    "A2": (-2, 0.5, CYAN, 0.32),
    "A3": (0, 2.8, GRN, 0.32),
    "A4": (1.5, -2.0, LBLUE, 0.32),
}

np.random.seed(42)
for name, (x, y, c, r) in agents.items():
    t = np.linspace(-1.5, 0, 8)
    hx = x + t * (0.5 if name != "Ego" else 0) + np.random.randn(8) * 0.08
    hy = y + t * (-0.4 if name in ["A3", "A4"] else 0.2) + np.random.randn(8) * 0.08
    ax.plot(hx, hy, color=c, lw=1.8, alpha=0.8, zorder=3)

for nb in ["A1", "A2", "A3", "A4"]:
    ex, ey = agents[nb][:2]
    ax.plot([0, ex], [0, ey], color=GRY, lw=1, alpha=0.5, ls="--", zorder=2)

for ang, c in zip([15, 5, -10], [ORG, "#FFA500", "#FFD700"]):
    r_ang = np.radians(ang + 90)
    px = np.linspace(0, 2.5 * np.cos(r_ang), 10)
    py = np.linspace(0, 2.5 * np.sin(r_ang), 10)
    ax.plot(px, py, color=c, lw=2, ls=":", alpha=0.9, zorder=4)

for name, (x, y, c, r) in agents.items():
    ax.add_patch(Circle((x, y), r, color=c, zorder=5, alpha=0.25))
    draw_vehicle(ax, x, y, c, scale=1.35 if name == "Ego" else 1.0, zorder=6)
    ax.text(
        x,
        y - r - 0.28,
        name,
        color=c,
        fontsize=7.5,
        ha="center",
        fontweight="bold",
        zorder=6,
    )

ax.text(
    0,
    3.6,
    "Bird's Eye View  -  GTNet Graph",
    color=WHT,
    fontsize=12,
    ha="center",
    fontweight="bold",
)
leg = [
    mpatches.Patch(color=ORG, label="Ego vehicle"),
    mpatches.Patch(color=BLUE, label="Surrounding agents"),
    plt.Line2D([0], [0], color=ORG, ls=":", label="Predicted trajectories (multi-mode)"),
    plt.Line2D([0], [0], color=CYAN, ls="-", label="History trajectory"),
]
ax.legend(handles=leg, loc="lower left", fontsize=7.5, facecolor=CARD, edgecolor=GRY, labelcolor=WHT)
save("fig1_bev")


# FIG 2 - Autonomous Driving Pipeline
f, ax = fig(12, 4)
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)

steps = [
    ("Sensors\n(Cam/LiDAR)", 0.7, CARD, BLUE),
    ("Perception\n(Detect/Track)", 2.7, CARD, CYAN),
    ("GTNet\nPrediction", 4.7, "#1A3550", ORG),
    ("Planning", 7.0, CARD, GRN),
    ("Control", 9.1, CARD, LBLUE),
]

for label, x, bg, border in steps:
    highlight = border == ORG
    lw = 3 if highlight else 1.5
    box = FancyBboxPatch((x, 1.2), 1.7, 1.6, boxstyle="round,pad=0.12", fc=bg, ec=border, lw=lw, zorder=2)
    ax.add_patch(box)
    ax.text(
        x + 0.85,
        2.0,
        label,
        color=WHT if not highlight else ORG,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold" if highlight else "normal",
        zorder=3,
    )

for x in [2.37, 4.37, 6.57, 8.67]:
    ax.annotate("", xy=(x + 0.33, 2.0), xytext=(x, 2.0), arrowprops=dict(arrowstyle="->", color=GRY, lw=2), zorder=3)

ax.annotate(
    "",
    xy=(6.57, 0.7),
    xytext=(4.7, 0.7),
    arrowprops=dict(arrowstyle="<->", color=ORG, lw=1.5, connectionstyle="arc3,rad=0"),
)
ax.text(5.63, 0.45, "Trajectory History -> Future Predictions", color=ORG, fontsize=7.5, ha="center")

ax.text(6, 3.7, "Autonomous Driving Pipeline", color=WHT, fontsize=13, ha="center", fontweight="bold")
save("fig2_pipeline")


# FIG 3 - Method Comparison Table
f, ax = fig(10, 4.5)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)

headers = ["Phương pháp", "Mô hình hoá\ntương tác", "Học từ\ndữ liệu", "Tính\nlinh hoạt"]
methods = [
    ("Single-Agent", "x", "✓", "x"),
    ("Social Force", "✓ (vật lý)", "x", "x"),
    ("GTNet (chúng tôi)", "✓ (đồ thị)", "✓", "✓"),
]
col_x = [1.0, 3.5, 6.0, 8.2]
row_y = [3.4, 2.4, 1.4, 0.4]

for hdr, cx in zip(headers, col_x):
    ax.text(cx, row_y[0], hdr, color=CYAN, fontsize=9.5, ha="center", va="center", fontweight="bold")

ax.axhline(3.0, color=GRY, lw=1, xmin=0.02, xmax=0.98, alpha=0.5)

for i, (method, *vals) in enumerate(methods):
    y = row_y[i + 1]
    bg_col = "#1A3550" if i == 2 else CARD
    box = FancyBboxPatch(
        (0.1, y - 0.38),
        9.8,
        0.76,
        boxstyle="round,pad=0.06",
        fc=bg_col,
        ec=ORG if i == 2 else GRY,
        lw=1.5 if i == 2 else 0.5,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        col_x[0],
        y,
        method,
        color=ORG if i == 2 else WHT,
        fontsize=9,
        ha="center",
        va="center",
        fontweight="bold" if i == 2 else "normal",
        zorder=3,
    )
    for val, cx in zip(vals, col_x[1:]):
        color = GRN if "✓" in val else RED if "x" in val else LBLUE
        ax.text(cx, y, val, color=color, fontsize=10, ha="center", va="center", zorder=3)

ax.text(5, 4.2, "So sánh phương pháp dự đoán quỹ đạo", color=WHT, fontsize=12, ha="center", fontweight="bold")
save("fig3_compare")


# FIG 4 - Problem Formulation
f, ax = fig(11, 5)
ax.set_xlim(-1, 11)
ax.set_ylim(0, 5)

ax.annotate("", xy=(10.5, 1.5), xytext=(0, 1.5), arrowprops=dict(arrowstyle="->", color=GRY, lw=2))
ax.text(10.6, 1.5, "t", color=GRY, fontsize=11, va="center")

ax.axvspan(0.2, 4.2, ymin=0.1, ymax=0.9, alpha=0.15, color=BLUE)
ax.text(2.2, 4.5, "Lịch sử (N_obs bước)\n≈ 2 giây", color=BLUE, fontsize=9, ha="center", fontweight="bold")

ax.axvspan(4.5, 9.5, ymin=0.1, ymax=0.9, alpha=0.15, color=ORG)
ax.text(
    7.0,
    4.5,
    "Tương lai (N_pred bước)\nK=3 quỹ đạo đa phương thức",
    color=ORG,
    fontsize=9,
    ha="center",
    fontweight="bold",
)

t_obs = np.linspace(0.5, 4.2, 15)
y_obs = 2.5 + 0.3 * np.sin(t_obs)
ax.plot(t_obs, y_obs, color=CYAN, lw=2.5, zorder=3, label="Lịch sử quan sát")
ax.scatter([4.2], [y_obs[-1]], color=CYAN, s=60, zorder=4)

angles = [0.25, 0.0, -0.3]
cols = [ORG, "#FF9F43", "#FFF200"]
labels = ["Mode 1 (rẽ trái)", "Mode 2 (đi thẳng)", "Mode 3 (rẽ phải)"]
for ang, c, lbl in zip(angles, cols, labels):
    t_pred = np.linspace(4.2, 9.2, 20)
    y_pred = y_obs[-1] + np.linspace(0, 5.0 * ang, 20) + 0.05 * np.random.randn(20)
    ax.plot(t_pred, y_pred, color=c, lw=2, ls="--", zorder=3, label=lbl)

ax.axvline(4.35, color=WHT, lw=1.5, ls=":", alpha=0.6)
ax.text(4.35, 0.7, "Hiện tại", color=WHT, fontsize=8, ha="center", alpha=0.8)

ax.legend(loc="lower right", fontsize=7.5, facecolor=CARD, edgecolor=GRY, labelcolor=WHT)
ax.text(5, 4.95, "Phát biểu bài toán: Input -> K Predicted Trajectories", color=WHT, fontsize=11, ha="center", fontweight="bold")
save("fig4_problem")


# FIG 5 - GTNet Baseline Architecture
f, ax = fig(12, 5)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)

blocks = [
    (1.0, 2.0, 2.0, 1.2, "Trajectory\nHistory\n(N×T×2)", BLUE),
    (3.5, 2.0, 2.0, 1.2, "GRU\nEncoder\n(per agent)", CYAN),
    (6.0, 2.0, 2.0, 1.2, "GCN\nMean Aggr.\n(graph)", GRN),
    (8.5, 2.0, 2.0, 1.2, "GRU\nDecoder\n(K=1)", ORG),
    (10.8, 2.3, 0.9, 0.7, "Smooth\nL1 Loss", RED),
]

for (x, y, w, h, lbl, c) in blocks:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=CARD, ec=c, lw=2.5, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, lbl, color=c, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=3)

for xs, xe in [(3.0, 3.5), (5.5, 6.0), (8.0, 8.5), (10.5, 10.8)]:
    ax.annotate("", xy=(xe, 2.6), xytext=(xs, 2.6), arrowprops=dict(arrowstyle="->", color=WHT, lw=1.8), zorder=3)

ax.text(4.5, 1.6, "h_i\n(hidden state)", color=CYAN, fontsize=7.5, ha="center")
ax.text(7.0, 1.6, "h_i' (aggregated)", color=GRN, fontsize=7.5, ha="center")
ax.text(9.5, 1.6, "ŷ (K=1)", color=ORG, fontsize=7.5, ha="center")

ax.text(6, 4.7, "GTNet Baseline Architecture", color=WHT, fontsize=12, ha="center", fontweight="bold")
save("fig5_baseline")


# FIG 7 - Mean Regression Problem
f, ax = fig(10, 6)
ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)

ax.fill_between([-4, 4], -0.8, 0.8, color="#1E3040", zorder=0)
ax.fill_betweenx([-3, 3], -0.8, 0.8, color="#1E3040", zorder=0)
for y in [-0.06, 0.06]:
    ax.axhline(y, color="#2A4A60", lw=0.6, ls="--")
for x in [-0.06, 0.06]:
    ax.axvline(x, color="#2A4A60", lw=0.6, ls="--")

ax.annotate("", xy=(0, 0), xytext=(-2.5, 0), arrowprops=dict(arrowstyle="->", color=CYAN, lw=2))
ax.text(-1.25, 0.4, "Lịch sử", color=CYAN, fontsize=8, ha="center")

real_paths = [
    ([0, 0, 1.5], [0, 2.0, 2.8], GRN, "GT: Rẽ trái"),
    ([0, 2.5, 3.5], [0, 0, 0], GRN, "GT: Đi thẳng"),
    ([0, 0, -1.5], [0, -2.0, -2.8], GRN, "GT: Rẽ phải"),
]
for xs, ys, c, lbl in real_paths:
    ax.plot(xs, ys, color=c, lw=2, ls="-", alpha=0.7, label=lbl)
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), arrowprops=dict(arrowstyle="->", color=c, lw=1.5))

ax.plot([0, 2.0, 3.0], [0, 0.8, 1.2], color=RED, lw=2.5, ls="--", zorder=4, label='Baseline: "Trung bình" (sai!)')
ax.annotate("", xy=(3.0, 1.2), xytext=(2.0, 0.8), arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.text(
    2.8,
    1.5,
    "Không\ncó thật!",
    color=RED,
    fontsize=8,
    ha="center",
    fontweight="bold",
    bbox=dict(fc=CARD, ec=RED, pad=3, boxstyle="round"),
)

ax.legend(loc="lower right", fontsize=7.5, facecolor=CARD, edgecolor=GRY, labelcolor=WHT)
ax.text(0, 2.8, "Mean Regression Problem tại giao lộ", color=WHT, fontsize=11, ha="center", fontweight="bold")
save("fig7_mean_regression")


# FIG 9 - ADE / FDE
f, ax = fig(10, 5)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1, 5)

t = np.linspace(0, 9, 12)
y_gt = 2.0 + 0.6 * np.sin(t * 0.5)
y_pred = 2.0 + 0.3 * np.sin(t * 0.5 + 0.4) + 0.3

ax.plot(t, y_gt, color=GRN, lw=2.5, label="Ground Truth", zorder=3)
ax.plot(t, y_pred, color=BLUE, lw=2.5, ls="--", label="Prediction", zorder=3)

for i in range(1, len(t) - 1, 2):
    ax.annotate("", xy=(t[i], y_gt[i]), xytext=(t[i], y_pred[i]), arrowprops=dict(arrowstyle="<->", color=ORG, lw=1.5))

ax.annotate("", xy=(t[-1], y_gt[-1]), xytext=(t[-1], y_pred[-1]), arrowprops=dict(arrowstyle="<->", color=RED, lw=2.5))
ax.text(t[-1] + 0.15, (y_gt[-1] + y_pred[-1]) / 2, "FDE", color=RED, fontsize=10, fontweight="bold", va="center")

ax.text(
    5.5,
    0.6,
    "ADE = avg over all timesteps",
    color=ORG,
    fontsize=9,
    ha="center",
    bbox=dict(fc=CARD, ec=ORG, pad=4, boxstyle="round"),
)

ax.scatter(t[[0, -1]], y_gt[[0, -1]], color=GRN, s=60, zorder=5)
ax.scatter(t[[0, -1]], y_pred[[0, -1]], color=BLUE, s=60, zorder=5)

ax.legend(loc="upper left", fontsize=9, facecolor=CARD, edgecolor=GRY, labelcolor=WHT)
ax.text(5, 4.7, "ADE & FDE - Trajectory Prediction Metrics", color=WHT, fontsize=11, ha="center", fontweight="bold")
save("fig9_ade_fde")


# FIG 11 - 3 Improvements Overview
f, ax = fig(12, 5)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)

cards = [
    (
        0.5,
        1.0,
        3.2,
        3.0,
        "Cải tiến 1\nGraph Attention\nNetwork (GAT)",
        "Học attention weight\ncó chọn lọc\nthay vì mean avg",
        BLUE,
    ),
    (
        4.4,
        1.0,
        3.2,
        3.0,
        "Cải tiến 2\nWinner-Takes-All\nLoss (WTA)",
        "K=3 modes,\nchỉ phạt mode\ngần GT nhất",
        ORG,
    ),
    (
        8.3,
        1.0,
        3.2,
        3.0,
        "Cải tiến 3\nAdaptive\nRadius",
        "Bán kính tương tác\nthay đổi theo\nvận tốc tác nhân",
        CYAN,
    ),
]

for (x, y, w, h, title, desc, c) in cards:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", fc=CARD, ec=c, lw=2.5, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.6, title, color=c, ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=3)
    ax.axhline(y + h - 1.1, xmin=x / 12, xmax=(x + w) / 12, color=c, lw=1, alpha=0.5)
    ax.text(x + w / 2, y + 0.7, desc, color=WHT, ha="center", va="center", fontsize=8.5, zorder=3)

problems = ["Mean Aggregation", "Mode Collapse", "Fixed Graph Radius"]
fixes = [BLUE, ORG, CYAN]
for i, (prob, c) in enumerate(zip(problems, fixes)):
    bx = 0.5 + i * 3.9 + 1.6
    ax.annotate("", xy=(bx, 1.0), xytext=(bx, 0.5), arrowprops=dict(arrowstyle="->", color=c, lw=1.8))
    ax.text(bx, 0.25, f"Giải quyết:\n{prob}", color=c, fontsize=7, ha="center", va="center")

ax.text(6, 4.7, "3 Cải tiến GTNet - Mỗi cải tiến giải quyết một điểm yếu", color=WHT, fontsize=11, ha="center", fontweight="bold")
save("fig11_improvements")


# FIG 12 - GCN vs GAT
f, ax = fig(12, 6)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)


def draw_graph(ax, cx, cy, title, c_edge, weights=None):
    positions = [
        (cx, cy + 1.5),
        (cx - 1.4, cy - 0.3),
        (cx + 1.4, cy - 0.3),
        (cx - 0.7, cy - 1.8),
        (cx + 0.7, cy - 1.8),
    ]
    center = positions[0]
    for i, (px, py) in enumerate(positions[1:], 1):
        w = weights[i - 1] if weights else 0.5
        lw = 1.0 + 3.5 * w if weights else 2
        alpha = 0.4 + 0.6 * w if weights else 0.7
        ax.plot([center[0], px], [center[1], py], color=c_edge, lw=lw, alpha=alpha, zorder=2)
        label = f"α={w:.2f}" if weights else "1/N"
        ax.text((center[0] + px) / 2 + 0.05, (center[1] + py) / 2 + 0.1, label, color=c_edge if weights else GRY, fontsize=7, ha="center", zorder=4)
    for idx, (px, py) in enumerate(positions):
        ax.add_patch(Circle((px, py), 0.35, color=CARD, ec=c_edge, lw=2, zorder=3))
        draw_vehicle(ax, px, py, ORG if idx == 0 else c_edge, scale=0.95, zorder=4)
    ax.text(cx, cy + 2.3, title, color=c_edge, fontsize=11, ha="center", fontweight="bold")


draw_graph(ax, 3.0, 2.8, "GCN  -  Mean Aggregation", GRY)
ax.text(3.0, 0.5, "Mọi hàng xóm trọng số = nhau", color=GRY, fontsize=8.5, ha="center", bbox=dict(fc=CARD, ec=GRY, pad=3, boxstyle="round"))

draw_graph(ax, 9.0, 2.8, "GAT  -  Attention Weights", ORG, weights=[0.62, 0.15, 0.08, 0.15])
ax.text(
    9.0,
    0.5,
    "Attention score α_ij học được\n(xe nguy hiểm = trọng số cao hơn)",
    color=ORG,
    fontsize=8.5,
    ha="center",
    bbox=dict(fc=CARD, ec=ORG, pad=3, boxstyle="round"),
)

ax.axvline(6.0, color=GRY, lw=1.5, ls="--", alpha=0.4)
ax.annotate("", xy=(7.0, 3.0), xytext=(5.0, 3.0), arrowprops=dict(arrowstyle="->", color=WHT, lw=2.5))
ax.text(6.0, 3.4, "Nâng cấp", color=WHT, fontsize=9, ha="center")

ax.text(6, 5.7, "GCN vs GAT: Cách tổng hợp thông tin đồ thị", color=WHT, fontsize=11, ha="center", fontweight="bold")
save("fig12_gcn_gat")


# FIG 13 - WTA Loss
f, ax = fig(10, 6)
ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)

ax.fill_between([-4, 4], -0.7, 0.7, color="#1E3040", zorder=0)
ax.fill_betweenx([-3, 3], -0.7, 0.7, color="#1E3040", zorder=0)

ax.annotate("", xy=(0, 0), xytext=(-2.5, 0), arrowprops=dict(arrowstyle="->", color=CYAN, lw=2.5))

modes = [
    ([0, 0.5, 0.5], [0, 1.8, 3.0], GRN, "Mode 1 (WINNER ★)", True),
    ([0, 2.5, 3.5], [0, -0.2, -0.2], ORG, "Mode 2", False),
    ([0, 0.3, 0.3], [0, -1.5, -2.8], LBLUE, "Mode 3", False),
]

gt = ([0, 0.6, 0.6], [0, 1.6, 2.8])
ax.plot(gt[0], gt[1], color=WHT, lw=2.5, ls=":", label="Ground Truth", zorder=5)

for xs, ys, c, lbl, is_winner in modes:
    lw = 3.0 if is_winner else 1.5
    ls = "-" if is_winner else "--"
    alpha = 1.0 if is_winner else 0.5
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, alpha=alpha, label=lbl, zorder=4)
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), arrowprops=dict(arrowstyle="->", color=c, lw=lw * 0.8, alpha=alpha))
    if is_winner:
        ax.text(xs[-1] + 0.2, ys[-1], "<- gradient\nback-prop", color=GRN, fontsize=8, va="center", fontweight="bold")
    else:
        ax.text(xs[-1] + 0.2, ys[-1], "x frozen", color=GRY, fontsize=8, va="center", alpha=0.7)

ax.legend(loc="lower right", fontsize=8, facecolor=CARD, edgecolor=GRY, labelcolor=WHT)
ax.text(0, 2.8, "Winner-Takes-All Loss  -  K=3 Modes", color=WHT, fontsize=11, ha="center", fontweight="bold")
save("fig13_wta")


# FIG 14 - Adaptive Radius
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
f.patch.set_facecolor(BG)
for ax in [ax1, ax2]:
    ax.set_facecolor(BG)
    ax.axis("off")


def draw_radius_scene(ax, r, speed_label, speed_color, title):
    ax.set_xlim(-r * 1.5 - 1, r * 1.5 + 1)
    ax.set_ylim(-r * 1.5 - 0.5, r * 1.5 + 1)
    ax.fill_betweenx([-r * 2, r * 2], -r * 1.5 - 1, r * 1.5 + 1, color="#1E3040", zorder=0)
    circ = Circle((0, 0), r, fill=False, ec=speed_color, lw=2.5, ls="--", alpha=0.7, zorder=2)
    ax.add_patch(circ)
    ax.text(r * 0.72, r * 0.72, f"r = {r:.0f}m", color=speed_color, fontsize=9, ha="center", zorder=4)
    draw_vehicle(ax, 0, 0, speed_color, scale=1.8, zorder=5)
    ax.annotate("", xy=(0, r * 0.6), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=speed_color, lw=2))
    ax.text(0.3, r * 0.3, speed_label, color=speed_color, fontsize=9, va="center", fontweight="bold")
    angles = [30, 120, 200, 310]
    distances = [r * 0.5, r * 0.7, r * 1.3, r * 0.9]
    for ang, dist in zip(angles, distances):
        a = np.radians(ang)
        x, y = dist * np.cos(a), dist * np.sin(a)
        inside = dist < r
        col = GRN if inside else RED
        alpha = 0.9 if inside else 0.4
        draw_vehicle(ax, x, y, col, scale=1.0, zorder=4, alpha=alpha)
        if inside:
            ax.plot([0, x], [0, y], color=GRN, lw=1.2, alpha=0.5, ls="--")
    ax.set_title(title, color=WHT, fontsize=10, fontweight="bold", pad=8)


draw_radius_scene(ax1, r=3.5, speed_label="v cao\n(highway)", speed_color=ORG, title="Tốc độ cao  ->  Bán kính lớn")
draw_radius_scene(ax2, r=1.5, speed_label="v thấp\n(city)", speed_color=CYAN, title="Tốc độ thấp  ->  Bán kính nhỏ")
plt.tight_layout(pad=0.5)
save("fig14_adaptive_radius")


# FIG 15 - Full GTNet Architecture
f, ax = fig(14, 5)
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)

blocks = [
    (0.3, 1.8, 1.8, 1.4, "Input\nTraj\nHistory", BLUE, False),
    (2.5, 1.8, 1.8, 1.4, "GRU\nEncoder\n(per agent)", CYAN, False),
    (4.8, 1.8, 2.2, 1.4, "GAT\n4 heads\nAdaptive r", ORG, True),
    (7.4, 1.8, 2.2, 1.4, "GRU\nDecoder\nK=3 modes", ORG, True),
    (10.0, 1.8, 1.8, 1.4, "WTA\nLoss\n(winner)", ORG, True),
    (12.2, 2.1, 1.5, 0.8, "minADE\nminFDE", GRN, False),
]

for (x, y, w, h, lbl, c, is_new) in blocks:
    ec = ORG if is_new else c
    lw = 3 if is_new else 2
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc="#1A3550" if is_new else CARD, ec=ec, lw=lw, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, lbl, color=ec, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=3)
    if is_new:
        ax.text(x + w / 2, y + h + 0.2, "★ New", color=ORG, fontsize=7, ha="center", zorder=3)

for xs, xe in [(2.1, 2.5), (4.3, 4.8), (7.0, 7.4), (9.6, 10.0), (11.8, 12.2)]:
    ax.annotate("", xy=(xe, 2.5), xytext=(xs, 2.5), arrowprops=dict(arrowstyle="->", color=WHT, lw=1.8))

ax.text(
    6.5,
    0.6,
    "Training: AdamW + CosineAnnealingLR + AMP (fp16) + Gradient Accumulation + Early Stopping",
    color=GRY,
    fontsize=8,
    ha="center",
    bbox=dict(fc=CARD, ec=GRY, pad=5, boxstyle="round", alpha=0.8),
)

ax.text(7, 4.7, "GTNet Full Architecture  -  ★ Thành phần mới (viền vàng)", color=WHT, fontsize=11, ha="center", fontweight="bold")
save("fig15_full_arch")


# FIG 17 - Ablation Study Table
f, ax = fig(12, 6)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)

headers = ["Biến thể", "GAT", "WTA", "Adaptive\nRadius", "minADE (m)", "minFDE (m)"]
rows = [
    ("Baseline  (000)", "x", "x", "x", "1.84", "4.12", False),
    ("GAT_only  (100)", "✓", "x", "x", "1.71", "3.80", False),
    ("Multi_only (010)", "x", "✓", "x", "1.45", "2.20", False),
    ("Radius_only (001)", "x", "x", "✓", "1.79", "4.05", False),
    ("GAT+Multi  (110)", "✓", "✓", "x", "1.38", "2.05", False),
    ("GAT+Radius (101)", "✓", "x", "✓", "1.65", "3.61", False),
    ("Multi+Radius (011)", "x", "✓", "✓", "1.40", "2.10", False),
    ("GTNet Full (111)", "✓", "✓", "✓", "1.28", "1.97", True),
]
col_x = [1.8, 4.0, 5.2, 6.6, 8.6, 10.4]
base_y = 5.1

for i, hdr in enumerate(headers):
    ax.text(col_x[i], base_y, hdr, color=CYAN, fontsize=8.5, ha="center", va="center", fontweight="bold")
ax.axhline(4.75, color=GRY, lw=1, xmin=0.02, xmax=0.98, alpha=0.6)

for j, (name, gat, wta, ar, ade, fde, is_best) in enumerate(rows):
    y = 4.3 - j * 0.52
    bg = "#1A3550" if is_best else CARD
    ec = ORG if is_best else GRY
    lw = 2.5 if is_best else 0.5
    box = FancyBboxPatch((0.1, y - 0.22), 11.8, 0.44, boxstyle="round,pad=0.05", fc=bg, ec=ec, lw=lw, zorder=2)
    ax.add_patch(box)
    n_col = ORG if is_best else WHT
    ax.text(col_x[0], y, name, color=n_col, fontsize=8, ha="center", va="center", fontweight="bold" if is_best else "normal", zorder=3)
    for val, cx in zip([gat, wta, ar], col_x[1:4]):
        c = GRN if "✓" in val else RED
        ax.text(cx, y, val, color=c, fontsize=11, ha="center", va="center", zorder=3)
    for val, cx in zip([ade, fde], col_x[4:]):
        c = GRN if is_best else WHT
        ax.text(cx, y, val, color=c, fontsize=9, ha="center", va="center", fontweight="bold" if is_best else "normal", zorder=3)

ax.text(6, 5.75, "Ablation Study  -  8 Biến thể (2³ Combinations)", color=WHT, fontsize=11, ha="center", fontweight="bold")
save("fig17_ablation")


# FIG 20 - Comparison Bar Chart
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
f.patch.set_facecolor(BG)

variants = ["Baseline", "GAT\nonly", "WTA\nonly", "GAT+\nWTA", "GTNet\nFull"]
ade_vals = [1.84, 1.71, 1.45, 1.38, 1.28]
fde_vals = [4.12, 3.80, 2.20, 2.05, 1.97]
colors = [GRY, BLUE, ORG, CYAN, GRN]

for ax, vals, metric, unit in [(ax1, ade_vals, "minADE", "m"), (ax2, fde_vals, "minFDE", "m")]:
    ax.set_facecolor(BG)
    bars = ax.bar(variants, vals, color=colors, width=0.6, zorder=2, edgecolor=BG, linewidth=1.5)
    bars[-1].set_edgecolor(ORG)
    bars[-1].set_linewidth(3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.2f}", ha="center", va="bottom", color=WHT, fontsize=9, fontweight="bold")
    ax.set_ylabel(f"{metric} ({unit})", color=WHT, fontsize=10)
    ax.set_title(f"{metric} Comparison", color=WHT, fontsize=11, fontweight="bold")
    ax.tick_params(colors=WHT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRY)
    ax.tick_params(axis="y", colors=WHT)
    ax.tick_params(axis="x", colors=WHT)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", color=GRY, alpha=0.3, zorder=0)

impv = (1 - fde_vals[-1] / fde_vals[0]) * 100
ax2.text(4, fde_vals[-1] + 0.35, f"↓ {impv:.0f}% vs Baseline", color=GRN, fontsize=9, ha="center", fontweight="bold")

plt.tight_layout(pad=1.5)
save("fig20_comparison")

print(f"\nAll images generated in {OUTPUT_DIR}")
