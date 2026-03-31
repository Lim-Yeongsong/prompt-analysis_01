"""
Turn별 의미 변화 시각화
- 참가자별 semantic trajectory (PCA 2D 궤적)
- 의미 변화량 (turn-to-turn cosine distance) 추이
- Turn 1 대비 의미적 drift
- 참가자 간 히트맵 비교
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 ────────────────────────────────────────────────────────────
nanum = [f.fname for f in fm.fontManager.ttflist if 'Nanum' in f.name]
if nanum:
    plt.rcParams['font.family'] = fm.FontProperties(fname=nanum[0]).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ─────────────────────────────────────────────────────────
df = pd.read_csv('/home/user/prompt-analysis_01/prompt_260316.csv')
df = df[df['prompt_raw'].notna() & (df['turn'].notna()) & (df['turn'].str.strip() != '')]
df['turn_num'] = df['turn'].str.extract(r'(\d+)').astype(float)
df['geft_score'] = pd.to_numeric(df['geft_score'], errors='coerce')
df['geft_group'] = pd.cut(df['geft_score'], bins=[-1, 9, 18],
                           labels=['저창의(0-9)', '고창의(10-18)'])

# 3턴 이상 참가자만
turn_counts = df.groupby('participant_id')['turn_num'].count()
valid_pids = turn_counts[turn_counts >= 3].index.tolist()
df_v = df[df['participant_id'].isin(valid_pids)].copy()
print(f"분석 대상: {len(valid_pids)}명, {len(df_v)}개 프롬프트")

# ── TF-IDF 임베딩 (전체 corpus 기준) ───────────────────────────────────
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                        max_features=3000, sublinear_tf=True)
tfidf.fit(df['prompt_raw'].tolist())   # 전체 데이터로 fit

# ════════════════════════════════════════════════════════════════════════
# Fig A: 참가자별 Semantic Trajectory (PCA 2D)
# ════════════════════════════════════════════════════════════════════════
pca = PCA(n_components=2, random_state=42)
all_embs = tfidf.transform(df_v['prompt_raw'].tolist()).toarray()
pca.fit(all_embs)

palette_group = {'저창의(0-9)': '#e74c3c', '고창의(10-18)': '#2ecc71'}
n_cols = 4
n_rows = -(-len(valid_pids) // n_cols)   # ceiling div
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
axes_flat = axes.flat

for idx, pid in enumerate(sorted(valid_pids, key=lambda p: int(p[1:]))):
    ax = axes_flat[idx]
    pdata = df_v[df_v['participant_id'] == pid].sort_values('turn_num')
    embs = pca.transform(tfidf.transform(pdata['prompt_raw'].tolist()).toarray())

    geft = pdata['geft_score'].iloc[0]
    group = str(pdata['geft_group'].iloc[0])
    color = palette_group.get(group, 'steelblue')

    # 궤적 선
    ax.plot(embs[:, 0], embs[:, 1], '-', color=color, linewidth=1.5, alpha=0.6)

    # 각 턴 점 + 번호
    norm = Normalize(vmin=1, vmax=len(pdata))
    cmap = plt.cm.Blues if 'NanumGothicCoding' else plt.cm.Blues
    for i, (x, y) in enumerate(embs):
        alpha = 0.4 + 0.6 * (i / max(len(pdata) - 1, 1))
        ax.scatter(x, y, s=80, color=color, alpha=alpha,
                   zorder=3, edgecolors='white', linewidth=0.5)
        ax.annotate(f"T{int(pdata.iloc[i]['turn_num'])}",
                    (x, y), fontsize=7, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    # 화살표 (첫→마지막)
    if len(embs) >= 2:
        ax.annotate('', xy=embs[-1], xytext=embs[0],
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.5, connectionstyle='arc3,rad=0.2'))

    ax.set_title(f'{pid}  (GEFT={int(geft)}, {group})', fontsize=9)
    ax.set_xlabel('PC1', fontsize=7)
    ax.set_ylabel('PC2', fontsize=7)
    ax.tick_params(labelsize=6)

# 빈 서브플롯 숨기기
for idx in range(len(valid_pids), n_rows * n_cols):
    axes_flat[idx].set_visible(False)

# 범례
patches = [mpatches.Patch(color=v, label=k) for k, v in palette_group.items()]
fig.legend(handles=patches, loc='lower right', fontsize=9)
fig.suptitle('참가자별 Turn 간 의미 변화 궤적 (PCA 2D)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig4a_semantic_trajectory.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("fig4a_semantic_trajectory.png 저장 완료")


# ════════════════════════════════════════════════════════════════════════
# Fig B: Turn-to-turn 의미 변화량 (cosine distance = 1 - similarity)
# ════════════════════════════════════════════════════════════════════════
change_rows = []
drift_rows = []

for pid in valid_pids:
    pdata = df_v[df_v['participant_id'] == pid].sort_values('turn_num')
    embs = tfidf.transform(pdata['prompt_raw'].tolist()).toarray()
    geft = pdata['geft_score'].iloc[0]
    group = str(pdata['geft_group'].iloc[0])

    # 연속 turn 간 변화량
    for i in range(len(pdata) - 1):
        sim = cosine_similarity(embs[i:i+1], embs[i+1:i+2])[0][0]
        change_rows.append({
            'participant_id': pid,
            'step': f"T{int(pdata.iloc[i]['turn_num'])}→T{int(pdata.iloc[i+1]['turn_num'])}",
            'from_turn': int(pdata.iloc[i]['turn_num']),
            'cosine_dist': 1 - sim,
            'geft_score': geft,
            'geft_group': group,
        })

    # turn 1 대비 drift
    for i in range(len(pdata)):
        sim0 = cosine_similarity(embs[0:1], embs[i:i+1])[0][0]
        drift_rows.append({
            'participant_id': pid,
            'turn_num': int(pdata.iloc[i]['turn_num']),
            'drift_from_t1': 1 - sim0,
            'geft_score': geft,
            'geft_group': group,
        })

change_df = pd.DataFrame(change_rows)
drift_df = pd.DataFrame(drift_rows)

# --- Fig B1: 개인 프로파일 (turn별 변화량)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), sharey=False)
axes_flat = axes.flat

for idx, pid in enumerate(sorted(valid_pids, key=lambda p: int(p[1:]))):
    ax = axes_flat[idx]
    sub = change_df[change_df['participant_id'] == pid].sort_values('from_turn')
    geft = sub['geft_score'].iloc[0]
    group = sub['geft_group'].iloc[0]
    color = palette_group.get(str(group), 'steelblue')

    ax.bar(sub['step'], sub['cosine_dist'], color=color, alpha=0.8, edgecolor='white')
    ax.set_title(f'{pid}  GEFT={int(geft)}', fontsize=9)
    ax.set_ylabel('의미 변화량\n(cosine dist)', fontsize=7)
    ax.tick_params(axis='x', labelsize=6, rotation=45)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_ylim(0, 1)

for idx in range(len(valid_pids), n_rows * n_cols):
    axes_flat[idx].set_visible(False)

fig.suptitle('참가자별 연속 Turn 간 의미 변화량 (cosine distance)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig4b_change_per_participant.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("fig4b_change_per_participant.png 저장 완료")


# ════════════════════════════════════════════════════════════════════════
# Fig C: GEFT 그룹별 Turn 진행에 따른 drift (평균 ± std)
# ════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# C1: 연속 turn 변화량
group_change = change_df.groupby(['geft_group', 'from_turn'])['cosine_dist'].agg(['mean', 'std']).reset_index()

for group, gdf in group_change.groupby('geft_group'):
    color = palette_group.get(str(group), 'gray')
    axes[0].plot(gdf['from_turn'], gdf['mean'], 'o-', label=str(group),
                 color=color, linewidth=2, markersize=8)
    axes[0].fill_between(gdf['from_turn'],
                         gdf['mean'] - gdf['std'],
                         gdf['mean'] + gdf['std'],
                         alpha=0.15, color=color)

axes[0].set_xlabel('Turn 번호 (→ 다음 Turn)')
axes[0].set_ylabel('의미 변화량 (cosine distance)')
axes[0].set_title('GEFT 그룹별 연속 Turn 의미 변화량\n(mean ± std)')
axes[0].legend()
axes[0].set_ylim(0, 1)
axes[0].set_xticks(sorted(change_df['from_turn'].unique()))

# C2: Turn 1 대비 누적 drift
group_drift = drift_df.groupby(['geft_group', 'turn_num'])['drift_from_t1'].agg(['mean', 'std']).reset_index()

for group, gdf in group_drift.groupby('geft_group'):
    color = palette_group.get(str(group), 'gray')
    axes[1].plot(gdf['turn_num'], gdf['mean'], 's-', label=str(group),
                 color=color, linewidth=2, markersize=8)
    axes[1].fill_between(gdf['turn_num'],
                         gdf['mean'] - gdf['std'],
                         gdf['mean'] + gdf['std'],
                         alpha=0.15, color=color)

axes[1].set_xlabel('Turn 번호')
axes[1].set_ylabel('Turn 1 대비 의미 변화량 (drift)')
axes[1].set_title('GEFT 그룹별 Turn 1 기준 의미 누적 Drift\n(mean ± std)')
axes[1].legend()
axes[1].set_ylim(0, 1)
axes[1].set_xticks(sorted(drift_df['turn_num'].unique()))

plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig4c_group_drift.png', dpi=150)
plt.close()
print("fig4c_group_drift.png 저장 완료")


# ════════════════════════════════════════════════════════════════════════
# Fig D: 참가자 × Turn 코사인 유사도 히트맵
# ════════════════════════════════════════════════════════════════════════
# 각 참가자의 최대 턴 수
max_turn = int(df_v['turn_num'].max())

# 참가자 × Turn 행렬 (turn1 대비 similarity)
pivot_data = {}
for pid in sorted(valid_pids, key=lambda p: int(p[1:])):
    pdata = df_v[df_v['participant_id'] == pid].sort_values('turn_num')
    embs = tfidf.transform(pdata['prompt_raw'].tolist()).toarray()
    row = {}
    for i, turn in enumerate(pdata['turn_num'].tolist()):
        sim0 = cosine_similarity(embs[0:1], embs[i:i+1])[0][0]
        row[int(turn)] = round(1 - sim0, 3)
    pivot_data[pid] = row

heatmap_df = pd.DataFrame(pivot_data).T.sort_index()
heatmap_df = heatmap_df.reindex(columns=sorted(heatmap_df.columns))
heatmap_df.columns = [f'T{c}' for c in heatmap_df.columns]

# geft 순 정렬
geft_map = df_v.groupby('participant_id')['geft_score'].first()
heatmap_df['geft_score'] = geft_map
heatmap_df = heatmap_df.sort_values('geft_score', ascending=False)
geft_labels = heatmap_df['geft_score'].astype(int)
heatmap_df = heatmap_df.drop(columns='geft_score')

fig, ax = plt.subplots(figsize=(max(10, len(heatmap_df.columns) * 0.9),
                                 max(6, len(heatmap_df) * 0.5)))
mask = heatmap_df.isna()
sns.heatmap(heatmap_df, ax=ax, mask=mask,
            cmap='YlOrRd', vmin=0, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.4, linecolor='lightgray',
            cbar_kws={'label': 'cosine distance from T1'})

# y축 레이블에 geft score 추가
ax.set_yticklabels([f'{pid} (G={g})' for pid, g in geft_labels.items()],
                   fontsize=8)
ax.set_xlabel('Turn', fontsize=10)
ax.set_title('참가자 × Turn — Turn 1 대비 의미 변화량 (GEFT 높은 순 정렬)', fontsize=12)
plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig4d_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig4d_heatmap.png 저장 완료")


# ════════════════════════════════════════════════════════════════════════
# 수치 요약
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("의미 변화 분석 요약")
print("="*55)
g_stats = change_df.groupby('geft_group')['cosine_dist'].agg(['mean', 'std', 'median'])
print("\n연속 Turn 의미 변화량 (geft 그룹별):")
print(g_stats.round(3))

d_stats = drift_df[drift_df['turn_num'] >= 3].groupby('geft_group')['drift_from_t1'].agg(['mean', 'std'])
print("\nTurn 3 이후 누적 Drift (geft 그룹별):")
print(d_stats.round(3))

print("\n저장 파일:")
for f in ['fig4a_semantic_trajectory.png',
          'fig4b_change_per_participant.png',
          'fig4c_group_drift.png',
          'fig4d_heatmap.png']:
    print(f"  - {f}")
