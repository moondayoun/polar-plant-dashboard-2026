import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os

# 한글 폰트 등록
for p in fm.findSystemFonts():
    if "nanum" in p.lower() or "noto" in p.lower():
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# CSV 로드
env = pd.read_csv("/home/user/uploads/0_________________1_.csv", encoding="utf-8-sig")
growth = pd.read_csv("/home/user/uploads/1_______________________________________.csv", encoding="utf-8-sig")

# 열 표준화 반영 (std_cols 핵심 부분 재현)
growth = growth.copy()
growth.columns = [str(c).strip() for c in growth.columns]
# 생육 데이터 표준 이름 매핑
rename_g = {}
for c in growth.columns:
    k = c.lower().replace(" ", "")
    if "생중량" in k: rename_g[c] = "생중량"
    elif "잎" in k: rename_g[c] = "잎 수"
    elif "지상부" in k and "길이" in k: rename_g[c] = "지상부 길이"
    elif "지하부" in k and "길이" in k: rename_g[c] = "지하부 길이"
growth = growth.rename(columns=rename_g)
# numeric coercion
for c in growth.columns:
    if c in ("개체번호",):
        continue
    if not pd.api.types.is_numeric_dtype(growth[c]):
        num = pd.to_numeric(growth[c].astype(str).str.replace(",","")
            .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
        if num.notna().mean() >= 0.5:
            growth[c] = num

# 생중량: 이미 있으므로 그대로 사용
for col in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]:
    if col in growth.columns:
        growth[col] = pd.to_numeric(growth[col], errors="coerce")

w = growth["생중량"].dropna()
ids = growth.loc[w.index, "개체번호"].tolist()

# 컬러: 상위/하위 25% 강조
q1, q3 = w.quantile(0.25), w.quantile(0.75)
colors = []
for val in w.values:
    if val >= q3:
        colors.append("#d32f2f")   # 상위 - 빨강
    elif val <= q1:
        colors.append("#1976d2")   # 하위 - 파랑
    else:
        colors.append("#90caf9")   # 중간 - 연파랑

# 등록 폰트 중 한글 폰트 찾기
korean_fonts = [f for f in fm.fontManager.ttflist
                if any(x in f.name.lower() for x in ["nanum","noto","cjk","korean"])]
fname = korean_fonts[0].fname if korean_fonts else None
if fname:
    fp = fm.FontProperties(fname=fname)
else:
    fp = None

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("인천아라고등학교 극지식물 개체별 생육 분석", fontsize=15, fontweight="bold")

# --- 그래프 1: 개체별 평균 생중량 막대 (상위/하위 강조) ---
ax1 = axes[0]
bars = ax1.bar(range(len(w)), w.values, color=colors, edgecolor="white", linewidth=0.5)
ax1.set_title("개체별 생중량 (g) ☆ 상위·하위 25% 강조", fontsize=11)
ax1.set_xlabel("개체번호", fontsize=10)
ax1.set_ylabel("생중량 (g)", fontsize=10)
if fp:
    ax1.set_xticklabels([str(i) for i in ids], fontproperties=fp, rotation=45, fontsize=8)
else:
    ax1.set_xticklabels([str(i) for i in ids], rotation=45, fontsize=8)
ax1.axhline(w.mean(), color="#d32f2f", linestyle="--", linewidth=1.2, label=f"평균 {w.mean():.2f}g")
ax1.legend(fontsize=9)
ax1.text(0.02, 0.97, f"n={len(w)}\n평균 {w.mean():.2f}g\nSD {w.std():.2f}g\nCV {w.std()/w.mean()*100:.1f}%",
         transform=ax1.transAxes, va="top", ha="left", fontsize=9,
         bbox=dict(boxstyle="round", facecolor="#e3f2fd", alpha=0.8))

# --- 그래프 2: 생중량 분포 박스플롯 ---
ax2 = axes[1]
bp = ax2.boxplot(w.values, vert=True, patch_artist=True,
                 boxprops=dict(facecolor="#bbdefb", color="#1976d2"),
                 medianprops=dict(color="#d32f2f", linewidth=2),
                 whiskerprops=dict(color="#1976d2"),
                 capprops=dict(color="#1976d2"),
                 flierprops=dict(marker="o", markerfacecolor="#ff9800",
                                 markersize=5, alpha=0.7))
ax2.set_title("개체별 생중량 분포 (박스플롯)", fontsize=11)
ax2.set_ylabel("생중량 (g)", fontsize=10)
ax2.set_xticks([1])
ax2.set_xticklabels(["33개체"], fontsize=10)
ax2.text(0.02, 0.97, f"Q1={q1:.2f}g\nQ3={q3:.2f}g\n사분위범위 {q3-q1:.2f}g",
         transform=ax2.transAxes, va="top", ha="left", fontsize=9,
         bbox=dict(boxstyle="round", facecolor="#e8f5e9", alpha=0.8))

# --- 그래프 3: 잎 수 vs 생중량 산점도 + 추세선 ---
ax3 = axes[2]
leaf = growth["잎 수"].dropna()
common = w.index.intersection(leaf.index)
x = leaf.loc[common].values
y = w.loc[common].values
ax3.scatter(x, y, color="#1976d2", s=55, alpha=0.75, edgecolors="white", linewidth=0.5, zorder=3)
slope, intercept = np.polyfit(x, y, 1)
xs = np.array([x.min(), x.max()])
ax3.plot(xs, slope*xs + intercept, color="#d32f2f", linewidth=2, label="추세선")
from scipy import stats as sp_stats
r, _ = sp_stats.pearsonr(x, y)
ax3.set_title(f"잎 수 vs 생중량  (r = {r:.2f})", fontsize=11)
ax3.set_xlabel("잎 수 (장)", fontsize=10)
ax3.set_ylabel("생중량 (g)", fontsize=10)
ax3.legend(fontsize=9, loc="upper left")
ax3.text(0.02, 0.97, f"n={len(x)}\nr = {r:.2f}\n기울기 {slope:.3f}",
         transform=ax3.transAxes, va="top", ha="left", fontsize=9,
         bbox=dict(boxstyle="round", facecolor="#fff3e0", alpha=0.8))

for ax in axes:
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = "/home/user/outputs"
plt.savefig(os.path.join(out, "polar_growth_analysis.png"), dpi=130, bbox_inches="tight")
print("저장 완료:", os.path.join(out, "polar_growth_analysis.png"))
print(f"개체 수: {len(w)}, 평균 생중량: {w.mean():.2f}g, CV: {w.std()/w.mean()*100:.1f}%")
print(f"잎 수 vs 생중량 r = {r:.2f}, 기울기 = {slope:.3f}")
