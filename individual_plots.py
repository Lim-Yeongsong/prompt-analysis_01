"""
참가자 25명 개별 의미 변화량 그래프
- Turn 1 기준 누적 drift (bar)
- 연속 turn-to-turn 변화량 (line overlay)
- 1턴 참가자도 포함 (표시만 다르게)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 ────────────────────────────────────────────────────────────
nanum = [f.fname for f in fm.fontManager.ttflist if 'Nanum' in f.name]
if nanum:
    plt.rcParams['font.family'] = fm.FontProperties(fname=nanum[0]).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ── 팔레트 ──────────────────────────────────────────────────────────────
C_LOW   = '#E74C3C'
C_HIGH  = '#2980B9'
C_BAR_L = '#FADBD8'
C_BAR_H = '#D6EAF8'
DARK    = '#2C3E50'
GRAY    = '#95A5A6'

# ── 데이터 로드 ──────────────────────────────────────────────────────────
df = pd.read_csv('/home/user/prompt-analysis_01/prompt_260316.csv')
df['geft_score'] = pd.to_numeric(df['geft_score'], errors='coerce')
df['turn_num']   = pd.to_numeric(df['turn'].str.extract(r'(\d+)')[0], errors='coerce')
df_clean = df[df['prompt_raw'].notna() & df['turn_num'].notna()].copy()
df_clean['geft_group'] = pd.cut(df_clean['geft_score'], bins=[-1,9,18],
                                 labels=['저창의(0-9)', '고창의(10-18)'])

# ── TF-IDF ───────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                        max_features=3000, sublinear_tf=True)
all_embs = tfidf.fit_transform(df_clean['prompt_raw'].tolist()).toarray()

# ── 참가자 정렬 ───────────────────────────────────────────────────────────
pids = sorted(df_clean['participant_id'].unique(), key=lambda p: int(p[1:]))
N = len(pids)   # 25

# ── 레이아웃: 5행 × 5열 ─────────────────────────────────────────────────
N_COLS = 5
N_ROWS = -(-N // N_COLS)   # ceil(25/5) = 5

fig, axes = plt.subplots(N_ROWS, N_COLS,
                         figsize=(N_COLS * 4.2, N_ROWS * 3.8),
                         facecolor='#F4F6F7')
fig.subplots_adjust(hspace=0.55, wspace=0.35,
                    left=0.05, right=0.97, top=0.93, bottom=0.05)

# ── 공통 제목 ─────────────────────────────────────────────────────────────
fig.suptitle('참가자별 Turn 의미 변화량\n(막대: Turn 1 대비 누적 drift  /  선: 연속 turn 간 변화량)',
             fontsize=15, fontweight='bold', color=DARK, y=0.97)

# 범례 패치
leg_patches = [
    mpatches.Patch(facecolor=C_BAR_L, edgecolor=C_LOW,  label='저창의(0-9) — drift from T1'),
    mpatches.Patch(facecolor=C_BAR_H, edgecolor=C_HIGH, label='고창의(10-18) — drift from T1'),
    plt.Line2D([0],[0], color=C_LOW,  linewidth=2, linestyle='--', marker='o',
               markersize=5, label='저창의 — step change'),
    plt.Line2D([0],[0], color=C_HIGH, linewidth=2, linestyle='--', marker='o',
               markersize=5, label='고창의 — step change'),
]
fig.legend(handles=leg_patches, loc='upper right', fontsize=8.5,
           bbox_to_anchor=(0.97, 0.97), framealpha=0.9)

# ── 참가자별 서브플롯 ────────────────────────────────────────────────────
for idx, pid in enumerate(pids):
    row, col = divmod(idx, N_COLS)
    ax = axes[row][col]
    ax.set_facecolor('white')

    pdata  = df_clean[df_clean['participant_id']==pid].sort_values('turn_num')
    geft   = pdata['geft_score'].iloc[0]
    group  = str(pdata['geft_group'].iloc[0])
    is_low = group.startswith('저')
    c_line = C_LOW  if is_low else C_HIGH
    c_bar  = C_BAR_L if is_low else C_BAR_H
    c_edge = C_LOW  if is_low else C_HIGH

    turns = pdata['turn_num'].tolist()
    embs  = all_embs[df_clean.index.get_indexer(pdata.index)]

    # ── 1턴 참가자 ─────────────────────────────────────────────────────
    if len(pdata) == 1:
        ax.bar([turns[0]], [0], color=c_bar, edgecolor=c_edge, linewidth=1.2,
               width=0.5, alpha=0.8)
        ax.text(0.5, 0.5, '프롬프트 1개\n(변화량 없음)',
                ha='center', va='center', fontsize=9, color=GRAY,
                transform=ax.transAxes)
        ax.set_xlim(turns[0]-0.8, turns[0]+0.8)
        ax.set_ylim(0, 1)
        ax.set_xticks(turns)
        ax.set_xticklabels([f'T{int(t)}' for t in turns], fontsize=7)
    else:
        # drift from T1
        drift_vals = []
        for i in range(len(pdata)):
            sim0 = cosine_similarity(embs[0:1], embs[i:i+1])[0][0]
            drift_vals.append(1 - sim0)

        # step change (연속 turn 간)
        step_vals = []
        step_turns = []
        for i in range(len(pdata)-1):
            sim = cosine_similarity(embs[i:i+1], embs[i+1:i+2])[0][0]
            step_vals.append(1 - sim)
            step_turns.append(turns[i+1])   # 변화가 일어난 turn

        # 막대 (drift)
        bars = ax.bar(turns, drift_vals, color=c_bar, edgecolor=c_edge,
                      linewidth=1.2, width=0.55, alpha=0.85, zorder=2)

        # 값 라벨
        for bar, val in zip(bars, drift_vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=6.5, color=c_edge, fontweight='bold')

        # 선 (step change) — 오른쪽 y축
        ax2 = ax.twinx()
        ax2.plot(step_turns, step_vals, '--o', color=c_line,
                 linewidth=1.8, markersize=5, alpha=0.9, zorder=3)
        # 각 점 라벨
        for xt, yt in zip(step_turns, step_vals):
            ax2.text(xt, yt + 0.03, f'{yt:.2f}',
                     ha='center', va='bottom', fontsize=6, color=c_line)
        ax2.set_ylim(0, 1.15)
        ax2.set_ylabel('step change', fontsize=6.5, color=c_line, labelpad=2)
        ax2.tick_params(axis='y', labelsize=6, colors=c_line)
        ax2.spines['right'].set_color(c_line)

        ax.set_xticks(turns)
        ax.set_xticklabels([f'T{int(t)}' for t in turns], fontsize=7.5)
        ax.set_xlim(min(turns)-0.7, max(turns)+0.7)

    # ── 공통 설정 ────────────────────────────────────────────────────────
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('drift from T1', fontsize=7, color=DARK, labelpad=2)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)

    # 제목 (GEFT 색상 배경)
    bg_color = '#FDEDEC' if is_low else '#EBF5FB'
    ax.set_title(f'{pid}   GEFT {int(geft)}점  |  {group}',
                 fontsize=9, fontweight='bold', color=c_edge, pad=6,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, edgecolor=c_edge, linewidth=0.8))

# ── 빈 서브플롯 숨기기 ──────────────────────────────────────────────────
for idx in range(N, N_ROWS * N_COLS):
    row, col = divmod(idx, N_COLS)
    axes[row][col].set_visible(False)

# ── 저장 ─────────────────────────────────────────────────────────────────
out_path = '/home/user/prompt-analysis_01/fig5_individual_semantic_change.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#F4F6F7')
plt.close()
print(f"저장 완료: {out_path}")
