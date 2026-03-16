"""
분석적(GEFT≥14.5) vs 전체적(GEFT<14.5) 스타일 의미 변화 분석
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 ────────────────────────────────────────────────────────────
nanum = [f.fname for f in fm.fontManager.ttflist if 'Nanum' in f.name]
if nanum:
    plt.rcParams['font.family'] = fm.FontProperties(fname=nanum[0]).get_name()
plt.rcParams['axes.unicode_minus'] = False

C_A    = '#2980B9'   # 분석적
C_G    = '#E67E22'   # 전체적
C_A_L  = '#D6EAF8'
C_G_L  = '#FDEBD0'
DARK   = '#2C3E50'
GRAY   = '#95A5A6'

# ═══════════════════════════════════════════════════════════════════════════
# 데이터 준비
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv('/home/user/prompt-analysis_01/prompt_260316.csv')
df['geft_score'] = pd.to_numeric(df['geft_score'], errors='coerce')
df['turn_num']   = pd.to_numeric(df['turn'].str.extract(r'(\d+)')[0], errors='coerce')
df['prompt_len'] = df['prompt_raw'].str.len()
df_clean = df[df['prompt_raw'].notna() & df['turn_num'].notna()].copy()

df_clean['style'] = df_clean['geft_score'].apply(
    lambda x: '분석적(≥14.5)' if x >= 14.5 else '전체적(<14.5)')
df_clean['style_en'] = df_clean['geft_score'].apply(
    lambda x: 'analytic' if x >= 14.5 else 'global')

# TF-IDF
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                        max_features=3000, sublinear_tf=True)
all_embs = tfidf.fit_transform(df_clean['prompt_raw'].tolist()).toarray()

# 참가자 정렬
pids_all = sorted(df_clean['participant_id'].unique(), key=lambda p: int(p[1:]))

# 각 참가자별 drift / step change 계산
drift_rows  = []
step_rows   = []
summary_rows = []

for pid in pids_all:
    pdata  = df_clean[df_clean['participant_id']==pid].sort_values('turn_num')
    geft   = pdata['geft_score'].iloc[0]
    style  = pdata['style'].iloc[0]
    embs   = all_embs[df_clean.index.get_indexer(pdata.index)]
    turns  = pdata['turn_num'].tolist()

    drifts = []
    for i, t in enumerate(turns):
        sim0 = cosine_similarity(embs[0:1], embs[i:i+1])[0][0]
        d = 1 - sim0
        drift_rows.append({'pid': pid, 'turn_num': int(t),
                           'drift': d, 'geft': geft, 'style': style})
        drifts.append(d)

    steps = []
    for i in range(len(pdata)-1):
        sim = cosine_similarity(embs[i:i+1], embs[i+1:i+2])[0][0]
        s = 1 - sim
        step_rows.append({'pid': pid,
                          'from_turn': int(turns[i]), 'to_turn': int(turns[i+1]),
                          'step': s, 'geft': geft, 'style': style})
        steps.append(s)

    summary_rows.append({
        'pid': pid, 'geft': geft, 'style': style,
        'n_turns': len(pdata),
        'max_drift': max(drifts),
        'final_drift': drifts[-1],
        'mean_step': np.mean(steps) if steps else np.nan,
        'max_step': max(steps) if steps else np.nan,
        'std_step': np.std(steps) if steps else np.nan,
        'monotone': all(drifts[i] <= drifts[i+1] for i in range(len(drifts)-1)),
    })

drift_df   = pd.DataFrame(drift_rows)
step_df    = pd.DataFrame(step_rows)
summary_df = pd.DataFrame(summary_rows)

# 그룹별 기술 통계
grp_step  = step_df.groupby('style')['step'].agg(['mean','std','median'])
grp_drift = drift_df[drift_df['turn_num']>=2].groupby('style')['drift'].agg(['mean','std','median'])

# t-test
a_step = step_df[step_df['style']=='분석적(≥14.5)']['step']
g_step = step_df[step_df['style']=='전체적(<14.5)']['step']
t_s, p_s = stats.ttest_ind(a_step, g_step)
a_mstep = summary_df[summary_df['style']=='분석적(≥14.5)']['mean_step'].dropna()
g_mstep = summary_df[summary_df['style']=='전체적(<14.5)']['mean_step'].dropna()
t_ms, p_ms = stats.ttest_ind(a_mstep, g_mstep)
a_fd = summary_df[summary_df['style']=='분석적(≥14.5)']['final_drift'].dropna()
g_fd = summary_df[summary_df['style']=='전체적(<14.5)']['final_drift'].dropna()
t_fd, p_fd = stats.ttest_ind(a_fd, g_fd)

# 숫자 요약 출력
print("="*60)
print("분석적 vs 전체적 — 의미 변화량 통계")
print("="*60)
print("\n[연속 step change]")
print(grp_step.round(3))
print(f"\n  t-test: t={t_s:.3f}, p={p_s:.3f}")
print(f"  참가자 평균 기준: t={t_ms:.3f}, p={p_ms:.3f}")
print("\n[최종 turn drift]")
print(grp_drift.round(3))
print(f"\n  t-test (final drift): t={t_fd:.3f}, p={p_fd:.3f}")
print("\n[단조 증가 (계속 멀어지는) 비율]")
for s, g in summary_df.groupby('style'):
    if g['n_turns'].max() >= 2:
        mono = g[g['n_turns']>=2]['monotone'].mean()
        print(f"  {s}: {mono*100:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════
# Fig 1: 개인별 25개 서브플롯 (스타일별 색상 구분)
# ═══════════════════════════════════════════════════════════════════════════
N_COLS = 5
N_ROWS = -(-len(pids_all) // N_COLS)

fig1, axes1 = plt.subplots(N_ROWS, N_COLS,
                            figsize=(N_COLS*4.2, N_ROWS*3.8),
                            facecolor='#F4F6F7')
fig1.subplots_adjust(hspace=0.55, wspace=0.35,
                     left=0.05, right=0.97, top=0.93, bottom=0.05)
fig1.suptitle('참가자별 의미 변화량\n분析的(GEFT≥14.5) vs 全体的(GEFT<14.5)',
              fontsize=15, fontweight='bold', color=DARK, y=0.97)

leg_patches = [
    mpatches.Patch(facecolor=C_A_L, edgecolor=C_A, label='분석적(≥14.5) — drift from T1'),
    mpatches.Patch(facecolor=C_G_L, edgecolor=C_G, label='전체적(<14.5) — drift from T1'),
    plt.Line2D([0],[0], color=C_A, lw=2, ls='--', marker='o', ms=5, label='분석적 — step change'),
    plt.Line2D([0],[0], color=C_G, lw=2, ls='--', marker='o', ms=5, label='전체적 — step change'),
]
fig1.legend(handles=leg_patches, loc='upper right', fontsize=8.5,
            bbox_to_anchor=(0.97, 0.97), framealpha=0.9)

col_map = {'분석적(≥14.5)': C_A, '전체적(<14.5)': C_G}
col_bg  = {'분석적(≥14.5)': C_A_L, '전체적(<14.5)': C_G_L}

for idx, pid in enumerate(pids_all):
    row, col = divmod(idx, N_COLS)
    ax = axes1[row][col]
    ax.set_facecolor('white')

    pdata = df_clean[df_clean['participant_id']==pid].sort_values('turn_num')
    geft  = pdata['geft_score'].iloc[0]
    style = pdata['style'].iloc[0]
    c_l   = col_map[style]
    c_bg  = col_bg[style]
    turns = pdata['turn_num'].tolist()
    embs  = all_embs[df_clean.index.get_indexer(pdata.index)]

    if len(pdata) == 1:
        ax.bar([turns[0]], [0], color=c_bg, edgecolor=c_l, linewidth=1.2, width=0.5)
        ax.text(0.5, 0.5, '1턴\n(변화 없음)', ha='center', va='center',
                fontsize=9, color=GRAY, transform=ax.transAxes)
        ax.set_xlim(turns[0]-0.8, turns[0]+0.8)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(turns)
        ax.set_xticklabels([f'T{int(t)}' for t in turns], fontsize=7.5)
    else:
        drift_vals = [1 - cosine_similarity(embs[0:1], embs[i:i+1])[0][0]
                      for i in range(len(pdata))]
        step_vals  = [1 - cosine_similarity(embs[i:i+1], embs[i+1:i+2])[0][0]
                      for i in range(len(pdata)-1)]
        step_turns = turns[1:]

        bars = ax.bar(turns, drift_vals, color=c_bg, edgecolor=c_l,
                      linewidth=1.2, width=0.55, alpha=0.9, zorder=2)
        for bar, val in zip(bars, drift_vals):
            if val > 0.04:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=6.5, color=c_l, fontweight='bold')

        ax2 = ax.twinx()
        ax2.plot(step_turns, step_vals, '--o', color=c_l,
                 linewidth=1.8, markersize=5, alpha=0.9, zorder=3)
        for xt, yt in zip(step_turns, step_vals):
            ax2.text(xt, yt+0.03, f'{yt:.2f}',
                     ha='center', va='bottom', fontsize=6, color=c_l)
        ax2.set_ylim(0, 1.15)
        ax2.set_ylabel('step', fontsize=6, color=c_l, labelpad=1)
        ax2.tick_params(axis='y', labelsize=5.5, colors=c_l)
        ax2.spines['right'].set_color(c_l)

        ax.set_xticks(turns)
        ax.set_xticklabels([f'T{int(t)}' for t in turns], fontsize=7.5)
        ax.set_xlim(min(turns)-0.7, max(turns)+0.7)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('drift from T1', fontsize=7, color=DARK, labelpad=1)
    ax.tick_params(axis='y', labelsize=6)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    short = '분석' if style.startswith('분') else '전체'
    ax.set_title(f'{pid}  GEFT {int(geft)}  [{short}]',
                 fontsize=9, fontweight='bold', color=c_l, pad=5,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=c_bg,
                           edgecolor=c_l, linewidth=0.8))

for idx in range(len(pids_all), N_ROWS*N_COLS):
    r, c = divmod(idx, N_COLS)
    axes1[r][c].set_visible(False)

plt.savefig('/home/user/prompt-analysis_01/fig6a_individual_by_style.png',
            dpi=150, bbox_inches='tight', facecolor='#F4F6F7')
plt.close()
print("\nfig6a_individual_by_style.png 저장 완료")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2: 그룹 비교 종합 (4패널)
# ═══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12), facecolor='#F4F6F7')
fig2.subplots_adjust(hspace=0.40, wspace=0.32,
                     left=0.07, right=0.96, top=0.90, bottom=0.08)
fig2.suptitle('분析的 vs 全体的 스타일 — 의미 변화 비교 분석\n(기준: GEFT 14.5점)',
              fontsize=16, fontweight='bold', color=DARK)

styles = ['분析적(≥14.5)', '전체적(<14.5)']   # 실제 값과 맞춤
styles = drift_df['style'].unique().tolist()

## ── 패널 1: Turn 진행별 누적 Drift (mean ± std) ──────────────────────────
ax = axes2[0][0]
ax.set_facecolor('white')
gd = drift_df.groupby(['style','turn_num'])['drift'].agg(['mean','std','count']).reset_index()
gd['se'] = gd['std'] / gd['count']**0.5

for sty, g in gd.groupby('style'):
    c = col_map[sty]
    ax.plot(g['turn_num'], g['mean'], 'o-', color=c, label=sty,
            linewidth=2.5, markersize=9, zorder=3)
    ax.fill_between(g['turn_num'], g['mean']-g['std'], g['mean']+g['std'],
                    alpha=0.15, color=c)
    # 각 지점 값 표시
    for _, row in g.iterrows():
        ax.text(row['turn_num'], row['mean']+0.04, f"{row['mean']:.2f}",
                ha='center', fontsize=7.5, color=c, fontweight='bold')

ax.set_xlabel('Turn 번호', fontsize=11)
ax.set_ylabel('drift from T1 (mean ± std)', fontsize=10)
ax.set_title('Turn 진행별 누적 의미 Drift', fontsize=12, fontweight='bold', pad=8)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xticks(sorted(drift_df['turn_num'].unique()))

## ── 패널 2: Step Change 분포 (violin + strip) ─────────────────────────────
ax = axes2[0][1]
ax.set_facecolor('white')
order = ['분析적(≥14.5)', '전체적(<14.5)']
order_real = sorted(styles, key=lambda s: 0 if s.startswith('분') else 1)

# violin
parts = ax.violinplot([step_df[step_df['style']==s]['step'].values for s in order_real],
                       positions=[1,2], widths=0.6, showmedians=True,
                       showextrema=True)
for i, (pc, s) in enumerate(zip(parts['bodies'], order_real)):
    pc.set_facecolor(col_map[s])
    pc.set_alpha(0.5)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)

# strip
for i, s in enumerate(order_real):
    data = step_df[step_df['style']==s]['step'].values
    jitter = np.random.normal(0, 0.04, size=len(data))
    ax.scatter(np.ones(len(data))*(i+1) + jitter, data,
               color=col_map[s], alpha=0.6, s=35, zorder=3, edgecolors='white', lw=0.5)

# 통계 표시
y_max = step_df['step'].max()
ax.plot([1,2], [y_max+0.06, y_max+0.06], 'k-', linewidth=1)
sig = '***' if p_s < 0.001 else ('**' if p_s < 0.01 else ('*' if p_s < 0.05 else f'p={p_s:.3f}'))
ax.text(1.5, y_max+0.08, sig, ha='center', fontsize=12, fontweight='bold')

ax.set_xticks([1,2])
ax.set_xticklabels(order_real, fontsize=10)
ax.set_ylabel('step change (cosine dist)', fontsize=10)
ax.set_title(f'연속 Turn 간 변화량 분포\n(t={t_s:.2f}, p={p_s:.3f})',
             fontsize=12, fontweight='bold', pad=8)
ax.set_ylim(0, 1.25)
ax.grid(axis='y', linestyle='--', alpha=0.3)

## ── 패널 3: 참가자 평균 Step Change (box + strip) ────────────────────────
ax = axes2[1][0]
ax.set_facecolor('white')

for i, s in enumerate(order_real):
    data = summary_df[summary_df['style']==s]['mean_step'].dropna().values
    bp = ax.boxplot(data, positions=[i+1], widths=0.4,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=2),
                    boxprops=dict(facecolor=col_map[s], alpha=0.5),
                    whiskerprops=dict(color=col_map[s]),
                    capprops=dict(color=col_map[s]))
    jitter = np.random.normal(0, 0.05, size=len(data))
    ax.scatter(np.ones(len(data))*(i+1)+jitter, data,
               color=col_map[s], s=70, zorder=3,
               edgecolors='white', linewidth=0.8, alpha=0.9)
    ax.text(i+1, data.mean()+0.03, f'μ={data.mean():.3f}',
            ha='center', fontsize=9, color=col_map[s], fontweight='bold')

sig2 = '***' if p_ms<0.001 else ('**' if p_ms<0.01 else ('*' if p_ms<0.05 else f'p={p_ms:.3f}'))
ax.plot([1,2],[0.87,0.87],'k-',lw=1)
ax.text(1.5,0.89,sig2,ha='center',fontsize=12,fontweight='bold')

ax.set_xticks([1,2])
ax.set_xticklabels(order_real, fontsize=10)
ax.set_ylabel('참가자 평균 step change', fontsize=10)
ax.set_title(f'참가자별 평균 변화량 비교\n(t={t_ms:.2f}, p={p_ms:.3f})',
             fontsize=12, fontweight='bold', pad=8)
ax.set_ylim(0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.3)

## ── 패널 4: 핵심 통계 테이블 ─────────────────────────────────────────────
ax = axes2[1][1]
ax.set_facecolor(DARK)
ax.axis('off')

metrics = [
    ('참가자 수', '14명', '11명'),
    ('평균 GEFT', '16.5점', '9.2점'),
    ('평균 Turn 수',
     f"{summary_df[summary_df['style'].str.startswith('분析적') | summary_df['style'].str.startswith('분析')]['n_turns'].mean():.1f}턴",
     ''),
]

# 동적으로 계산
an_sum = summary_df[summary_df['style']=='분析적(≥14.5)'] if '분析적(≥14.5)' in summary_df['style'].values else summary_df[summary_df['style']=='분析적(≥14.5)']
# 실제 style 값 사용
an_sum = summary_df[summary_df['style'].str.startswith('분')]
gl_sum = summary_df[summary_df['style'].str.startswith('전')]
an_step_g = step_df[step_df['style'].str.startswith('분')]
gl_step_g = step_df[step_df['style'].str.startswith('전')]

rows_data = [
    ('참가자 수',           f"{len(an_sum)}명",                       f"{len(gl_sum)}명"),
    ('평균 GEFT',           f"{an_sum['geft'].mean():.1f}점",          f"{gl_sum['geft'].mean():.1f}점"),
    ('평균 Turn 수',        f"{an_sum['n_turns'].mean():.1f}턴",       f"{gl_sum['n_turns'].mean():.1f}턴"),
    ('step change 평균',    f"{an_step_g['step'].mean():.3f}",         f"{gl_step_g['step'].mean():.3f}"),
    ('step change std',     f"{an_step_g['step'].std():.3f}",          f"{gl_step_g['step'].std():.3f}"),
    ('최종 drift 평균',     f"{an_sum['final_drift'].mean():.3f}",     f"{gl_sum['final_drift'].mean():.3f}"),
    ('최대 drift 평균',     f"{an_sum['max_drift'].mean():.3f}",       f"{gl_sum['max_drift'].mean():.3f}"),
    ('단조증가 비율',
     f"{an_sum[an_sum['n_turns']>=2]['monotone'].mean()*100:.0f}%",
     f"{gl_sum[gl_sum['n_turns']>=2]['monotone'].mean()*100:.0f}%"),
    ('t-test (step)',       f"t={t_s:.2f}",                            f"p={p_s:.3f}"),
    ('t-test (final drift)',f"t={t_fd:.2f}",                           f"p={p_fd:.3f}"),
]

ax.text(0.5, 0.97, '그룹별 핵심 통계 요약', ha='center', va='top',
        fontsize=13, fontweight='bold', color='white', transform=ax.transAxes)

# 헤더
col_xs = [0.08, 0.48, 0.75]
headers = ['지표', '분析的(≥14.5)', '全体的(<14.5)']
head_cols = ['#ECF0F1', C_A, C_G]
for cx, hd, hc in zip(col_xs, headers, head_cols):
    ax.text(cx, 0.89, hd, ha='left', va='top',
            fontsize=10, fontweight='bold', color=hc, transform=ax.transAxes)

ax.plot([0.04, 0.97], [0.86, 0.86], color='#5D6D7E',
        linewidth=0.8, transform=ax.transAxes)

y = 0.82
for label, an_val, gl_val in rows_data:
    ax.text(col_xs[0], y, label,  ha='left', va='top', fontsize=9,
            color='#BDC3C7', transform=ax.transAxes)
    ax.text(col_xs[1], y, an_val, ha='left', va='top', fontsize=9,
            color=C_A,       transform=ax.transAxes, fontweight='bold')
    ax.text(col_xs[2], y, gl_val, ha='left', va='top', fontsize=9,
            color=C_G,       transform=ax.transAxes, fontweight='bold')
    y -= 0.082
    ax.plot([0.04, 0.97], [y+0.01, y+0.01], color='#34495E',
            linewidth=0.4, transform=ax.transAxes)

plt.savefig('/home/user/prompt-analysis_01/fig6b_style_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='#F4F6F7')
plt.close()
print("fig6b_style_comparison.png 저장 완료")
