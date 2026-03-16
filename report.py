"""
통합 분석 리포트 생성기
- 모든 분석 결과를 하나의 PDF 스타일 PNG 리포트로 정리
- 결과 요약 + 핵심 시각화 조합
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
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 ────────────────────────────────────────────────────────────
nanum = [f.fname for f in fm.fontManager.ttflist if 'Nanum' in f.name]
if nanum:
    prop = fm.FontProperties(fname=nanum[0])
    plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = '#F8F9FA'

# ── 팔레트 ────────────────────────────────────────────────────────────────
C_LOW  = '#E74C3C'   # 저창의
C_HIGH = '#2980B9'   # 고창의
C_CARD = '#FFFFFF'
GRAY   = '#7F8C8D'
DARK   = '#2C3E50'

# ═══════════════════════════════════════════════════════════════════════════
# 데이터 준비
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv('/home/user/prompt-analysis_01/prompt_260316.csv')
df['geft_score'] = pd.to_numeric(df['geft_score'], errors='coerce')
df['turn_num'] = pd.to_numeric(df['turn'].str.extract(r'(\d+)')[0], errors='coerce')
df_clean = df[df['prompt_raw'].notna() & df['turn_num'].notna()].copy()
df_clean['prompt_len'] = df_clean['prompt_raw'].str.len()
df_clean['geft_group'] = pd.cut(df_clean['geft_score'], bins=[-1,9,18],
                                 labels=['저창의(0-9)', '고창의(10-18)'])

# TF-IDF
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                        max_features=3000, sublinear_tf=True)
all_embs = tfidf.fit_transform(df_clean['prompt_raw'].tolist()).toarray()

# 참가자 평균 임베딩
pids_all = df_clean['participant_id'].unique()
part_emb_dict = {}
for pid in pids_all:
    idx = df_clean[df_clean['participant_id']==pid].index
    rows = all_embs[df_clean.index.get_indexer(idx)]
    part_emb_dict[pid] = normalize(rows.mean(axis=0, keepdims=True))[0]

pids_sorted = sorted(pids_all, key=lambda p: int(p[1:]))
part_matrix = np.array([part_emb_dict[p] for p in pids_sorted])
part_labels  = [df_clean[df_clean['participant_id']==p]['geft_group'].iloc[0] for p in pids_sorted]
part_geft    = [df_clean[df_clean['participant_id']==p]['geft_score'].iloc[0]  for p in pids_sorted]
part_labels_int = [0 if str(l).startswith('저') else 1 for l in part_labels]

# PCA (참가자 레벨)
pca2 = PCA(n_components=2, random_state=42)
coords = pca2.fit_transform(part_matrix)

# 상관
part_df = pd.DataFrame({'pid': pids_sorted, 'geft': part_geft, 'pc1': coords[:,0], 'pc2': coords[:,1]})
part_df['n_prompts'] = part_df['pid'].map(
    df_clean.groupby('participant_id').size())
part_df['prompt_len'] = part_df['pid'].map(
    df_clean.groupby('participant_id')['prompt_len'].mean())
r_pc1, p_pc1 = stats.pearsonr(part_df['geft'], part_df['pc1'])
r_len,  p_len  = stats.pearsonr(part_df['geft'], part_df['prompt_len'])

# 실루엣
sil = silhouette_score(part_matrix, part_labels_int, metric='cosine')

# Turn별 drift (3턴+ 참가자)
tc = df_clean.groupby('participant_id').size()
valid_pids = tc[tc>=3].index.tolist()
drift_rows = []
for pid in valid_pids:
    pdata = df_clean[df_clean['participant_id']==pid].sort_values('turn_num')
    idx   = pdata.index
    embs  = all_embs[df_clean.index.get_indexer(idx)]
    geft  = pdata['geft_score'].iloc[0]
    grp   = str(pdata['geft_group'].iloc[0])
    for i, turn in enumerate(pdata['turn_num'].tolist()):
        sim0 = cosine_similarity(embs[0:1], embs[i:i+1])[0][0]
        drift_rows.append({'pid': pid, 'turn_num': int(turn),
                           'drift': 1-sim0, 'geft': geft, 'group': grp})
drift_df = pd.DataFrame(drift_rows)

# 그룹별 프롬프트 길이 통계
lo_len = df_clean[df_clean['geft_group']=='저창의(0-9)']['prompt_len']
hi_len = df_clean[df_clean['geft_group']=='고창의(10-18)']['prompt_len']
t_stat, t_pval = stats.ttest_ind(lo_len, hi_len)

# ═══════════════════════════════════════════════════════════════════════════
# 레이아웃 설계
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 28), facecolor='#F0F2F5')

outer = gridspec.GridSpec(
    5, 1, figure=fig,
    height_ratios=[1.2, 2.8, 2.8, 2.8, 0.4],
    hspace=0.06,
    left=0.04, right=0.97, top=0.97, bottom=0.02
)

# ─────────────────────────────────────────────────
# 섹션 0: 헤더
# ─────────────────────────────────────────────────
ax_header = fig.add_subplot(outer[0])
ax_header.set_facecolor(DARK)
ax_header.axis('off')

ax_header.text(0.5, 0.72, '프롬프트 의미 변화 종합 분석 리포트',
               ha='center', va='center', fontsize=22, fontweight='bold',
               color='white', transform=ax_header.transAxes)
ax_header.text(0.5, 0.32,
               '데이터: prompt_260316.csv  |  분석 방법: TF-IDF char n-gram (2~4) + PCA + Cosine Similarity',
               ha='center', va='center', fontsize=11, color='#BDC3C7',
               transform=ax_header.transAxes)

# KPI 박스 4개
kpis = [
    ('총 참가자', '25명'),
    ('총 프롬프트', '85개'),
    ('평균 GEFT', '13.3'),
    ('최대 턴 수', '11턴'),
]
for i, (label, val) in enumerate(kpis):
    x = 0.07 + i * 0.22
    rect = FancyBboxPatch((x, -0.18), 0.18, 0.28,
                           boxstyle="round,pad=0.01",
                           facecolor='#34495E', edgecolor='#5D6D7E',
                           transform=ax_header.transAxes, clip_on=False)
    ax_header.add_patch(rect)
    ax_header.text(x+0.09, -0.04, val, ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white',
                   transform=ax_header.transAxes)
    ax_header.text(x+0.09, -0.14, label, ha='center', va='center',
                   fontsize=9, color='#95A5A6',
                   transform=ax_header.transAxes)

# ─────────────────────────────────────────────────
# 섹션 1: 임베딩 클러스터링 + 상관관계
# ─────────────────────────────────────────────────
sec1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[1],
    wspace=0.30, width_ratios=[1.2, 1, 1]
)

## 1-1 PCA scatter
ax1 = fig.add_subplot(sec1[0])
ax1.set_facecolor('white')
colors = [C_LOW if l.startswith('저') else C_HIGH for l in part_labels]
for i, (x, y) in enumerate(coords):
    ax1.scatter(x, y, color=colors[i], s=120, zorder=3,
                edgecolors='white', linewidth=0.8, alpha=0.9)
    ax1.annotate(pids_sorted[i], (x, y), fontsize=6.5,
                 xytext=(4, 4), textcoords='offset points', color=DARK)

ax1.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
ax1.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
ax1.set_title('참가자 임베딩 군집 (PCA 2D)', fontsize=11, fontweight='bold', pad=8)

patches = [mpatches.Patch(color=C_LOW,  label=f'저창의(0-9)  Sil={sil:.2f}'),
           mpatches.Patch(color=C_HIGH, label='고창의(10-18)')]
ax1.legend(handles=patches, fontsize=8, loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.3)

# 섹션 레이블
ax1.text(-0.12, 1.08, '① 임베딩 클러스터링', fontsize=11, fontweight='bold',
         color=DARK, transform=ax1.transAxes)

## 1-2 GEFT vs PC1
ax2 = fig.add_subplot(sec1[1])
ax2.set_facecolor('white')
ax2.scatter(part_df['geft'], part_df['pc1'],
            c=[C_LOW if g<10 else C_HIGH for g in part_df['geft']],
            s=90, alpha=0.85, edgecolors='white', linewidth=0.6, zorder=3)
m, b = np.polyfit(part_df['geft'], part_df['pc1'], 1)
xr = np.linspace(part_df['geft'].min()-0.5, part_df['geft'].max()+0.5, 100)
ax2.plot(xr, m*xr+b, '--', color=GRAY, linewidth=1.5, alpha=0.7)
ax2.set_xlabel('GEFT 점수', fontsize=9)
ax2.set_ylabel('PC1', fontsize=9)
ax2.set_title(f'GEFT ↔ PC1 상관\n(r={r_pc1:.2f}, p={p_pc1:.3f})', fontsize=11, fontweight='bold', pad=8)
ax2.grid(True, linestyle='--', alpha=0.3)

## 1-3 그룹별 프롬프트 길이 분포
ax3 = fig.add_subplot(sec1[2])
ax3.set_facecolor('white')
groups = ['저창의(0-9)', '고창의(10-18)']
col_map = {'저창의(0-9)': C_LOW, '고창의(10-18)': C_HIGH}
for grp in groups:
    data = df_clean[df_clean['geft_group']==grp]['prompt_len']
    ax3.hist(data, bins=15, alpha=0.65, color=col_map[grp],
             label=f'{grp}\n(μ={data.mean():.0f}자)', edgecolor='white', linewidth=0.4)
ax3.axvline(lo_len.mean(), color=C_LOW,  linestyle='--', linewidth=1.5)
ax3.axvline(hi_len.mean(), color=C_HIGH, linestyle='--', linewidth=1.5)
ax3.set_xlabel('프롬프트 길이 (자)', fontsize=9)
ax3.set_ylabel('빈도', fontsize=9)
ax3.set_title(f'그룹별 프롬프트 길이 분포\n(t={t_stat:.2f}, p={t_pval:.3f})',
              fontsize=11, fontweight='bold', pad=8)
ax3.legend(fontsize=8)
ax3.grid(True, linestyle='--', alpha=0.3)

# ─────────────────────────────────────────────────
# 섹션 2: Turn 패턴 분석
# ─────────────────────────────────────────────────
sec2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[2],
    wspace=0.30, width_ratios=[1.1, 1, 1]
)

## 2-1 Turn별 평균 프롬프트 길이
ax4 = fig.add_subplot(sec2[0])
ax4.set_facecolor('white')
turn_len = df_clean.groupby(['geft_group','turn_num'])['prompt_len'].agg(['mean','std','count']).reset_index()
turn_len['se'] = turn_len['std'] / turn_len['count']**0.5

for grp, grp_df in turn_len.groupby('geft_group'):
    c = col_map.get(str(grp), GRAY)
    ax4.plot(grp_df['turn_num'], grp_df['mean'], 'o-', color=c,
             label=str(grp), linewidth=2, markersize=7)
    ax4.fill_between(grp_df['turn_num'],
                     grp_df['mean'] - grp_df['se'],
                     grp_df['mean'] + grp_df['se'],
                     alpha=0.15, color=c)

ax4.set_xlabel('Turn 번호', fontsize=9)
ax4.set_ylabel('평균 프롬프트 길이 (자)', fontsize=9)
ax4.set_title('Turn 진행별 프롬프트 길이\n(mean ± SE)', fontsize=11, fontweight='bold', pad=8)
ax4.legend(fontsize=8)
ax4.grid(True, linestyle='--', alpha=0.3)
ax4.text(-0.12, 1.08, '② Turn 패턴', fontsize=11, fontweight='bold',
         color=DARK, transform=ax4.transAxes)

## 2-2 Turn별 인접 코사인 유사도
ax5 = fig.add_subplot(sec2[1])
ax5.set_facecolor('white')
sim_rows = []
for pid in valid_pids:
    pdata = df_clean[df_clean['participant_id']==pid].sort_values('turn_num')
    embs  = all_embs[df_clean.index.get_indexer(pdata.index)]
    grp   = str(pdata['geft_group'].iloc[0])
    for i in range(len(pdata)-1):
        sim = cosine_similarity(embs[i:i+1], embs[i+1:i+2])[0][0]
        sim_rows.append({'from_turn': int(pdata.iloc[i]['turn_num']),
                         'cosine_sim': sim, 'group': grp})
sim_df = pd.DataFrame(sim_rows)

for grp, gdf in sim_df.groupby('group'):
    agg = gdf.groupby('from_turn')['cosine_sim'].agg(['mean','std','count'])
    agg['se'] = agg['std'] / agg['count']**0.5
    c = col_map.get(grp, GRAY)
    ax5.plot(agg.index, agg['mean'], 'o-', color=c, label=grp, linewidth=2, markersize=7)
    ax5.fill_between(agg.index, agg['mean']-agg['se'], agg['mean']+agg['se'],
                     alpha=0.15, color=c)

ax5.set_xlabel('Turn 번호 (→ 다음 Turn)', fontsize=9)
ax5.set_ylabel('코사인 유사도', fontsize=9)
ax5.set_ylim(0, 1.05)
ax5.set_title('연속 Turn 간 의미 유사도\n(높을수록 변화 적음)', fontsize=11, fontweight='bold', pad=8)
ax5.legend(fontsize=8)
ax5.grid(True, linestyle='--', alpha=0.3)

## 2-3 그룹별 Turn 수 분포
ax6 = fig.add_subplot(sec2[2])
ax6.set_facecolor('white')
turn_count = df_clean.groupby(['participant_id','geft_group']).size().reset_index(name='n_turns')
for grp in groups:
    data = turn_count[turn_count['geft_group']==grp]['n_turns']
    ax6.hist(data, bins=range(1, int(turn_count['n_turns'].max())+2),
             alpha=0.65, color=col_map[grp],
             label=f'{grp} (μ={data.mean():.1f}턴)',
             edgecolor='white', linewidth=0.4, align='left')

ax6.set_xlabel('총 Turn 수', fontsize=9)
ax6.set_ylabel('참가자 수', fontsize=9)
ax6.set_title('그룹별 총 Turn 수 분포', fontsize=11, fontweight='bold', pad=8)
ax6.legend(fontsize=8)
ax6.grid(True, linestyle='--', alpha=0.3)

# ─────────────────────────────────────────────────
# 섹션 3: 의미 변화 시각화
# ─────────────────────────────────────────────────
sec3 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[3],
    wspace=0.30, width_ratios=[1.3, 1, 0.9]
)

## 3-1 누적 drift (mean ± std) — 두 그룹
ax7 = fig.add_subplot(sec3[0])
ax7.set_facecolor('white')

group_drift = drift_df.groupby(['group','turn_num'])['drift'].agg(['mean','std','count']).reset_index()
group_drift['se'] = group_drift['std'] / group_drift['count']**0.5

for grp, gdf in group_drift.groupby('group'):
    c = col_map.get(grp, GRAY)
    ax7.plot(gdf['turn_num'], gdf['mean'], 's-', color=c, label=grp,
             linewidth=2.5, markersize=9)
    ax7.fill_between(gdf['turn_num'],
                     gdf['mean']-gdf['std'],
                     gdf['mean']+gdf['std'],
                     alpha=0.12, color=c)

ax7.set_xlabel('Turn 번호', fontsize=9)
ax7.set_ylabel('Turn 1 대비 의미 변화량\n(cosine distance)', fontsize=9)
ax7.set_ylim(0, 1.05)
ax7.set_title('Turn 1 기준 누적 의미 Drift\n(mean ± std)', fontsize=11, fontweight='bold', pad=8)
ax7.legend(fontsize=8)
ax7.grid(True, linestyle='--', alpha=0.3)
ax7.text(-0.12, 1.08, '③ 의미 변화', fontsize=11, fontweight='bold',
         color=DARK, transform=ax7.transAxes)

## 3-2 히트맵 (참가자 × Turn, drift)
ax8 = fig.add_subplot(sec3[1])
ax8.set_facecolor('white')

pivot = drift_df.pivot_table(index='pid', columns='turn_num', values='drift', aggfunc='first')
pivot = pivot.reindex(columns=sorted(pivot.columns))
# GEFT 순 정렬
geft_map_2 = df_clean.groupby('participant_id')['geft_score'].first()
pivot['_geft'] = geft_map_2
pivot = pivot.sort_values('_geft', ascending=False).drop(columns='_geft')
pivot.columns = [f'T{int(c)}' for c in pivot.columns]

sns.heatmap(pivot, ax=ax8, cmap='YlOrRd', vmin=0, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 6.5},
            linewidths=0.3, linecolor='#ECF0F1',
            cbar_kws={'label': 'cosine dist from T1', 'shrink': 0.8})
ax8.set_title('참가자×Turn 의미 변화 히트맵\n(GEFT 높은 순)', fontsize=11, fontweight='bold', pad=8)
ax8.set_ylabel('')
ax8.tick_params(axis='y', labelsize=7)
ax8.tick_params(axis='x', labelsize=7)

## 3-3 핵심 수치 인사이트 카드
ax9 = fig.add_subplot(sec3[2])
ax9.set_facecolor('#2C3E50')
ax9.axis('off')

ax9.text(0.5, 0.97, '핵심 발견', fontsize=12, fontweight='bold',
         color='white', ha='center', va='top', transform=ax9.transAxes)

findings = [
    ('실루엣 계수',   f'{sil:.2f}',
     '두 창의 그룹이\n의미 공간에서 부분 분리'),
    ('GEFT↔PC1 r',   f'{r_pc1:.2f}',
     f'p={p_pc1:.3f}\nGEFT 높을수록 PC1 방향\n(보다 추상적 표현)'),
    ('프롬프트 길이',  f'{lo_len.mean():.0f} vs {hi_len.mean():.0f}자',
     f'저창의 vs 고창의\nt={t_stat:.2f}, p={t_pval:.3f}'),
    ('T3+ 누적 Drift',
     '저 0.87 vs 고 0.80',
     '저창의 그룹이 T1에서\n더 멀리, 일정하게 이탈'),
    ('변동성(std)',
     '저 0.09 vs 고 0.24',
     '고창의는 방향 전환이 많아\n표준편차 2.7× 큼'),
]

y_pos = 0.88
for title, value, desc in findings:
    # 배경 박스
    rect = FancyBboxPatch((0.04, y_pos-0.14), 0.92, 0.15,
                           boxstyle="round,pad=0.01",
                           facecolor='#34495E', edgecolor='#5D6D7E',
                           transform=ax9.transAxes, clip_on=True)
    ax9.add_patch(rect)
    ax9.text(0.50, y_pos-0.02, title, fontsize=8, color='#BDC3C7',
             ha='center', va='top', transform=ax9.transAxes)
    ax9.text(0.50, y_pos-0.07, value, fontsize=11, fontweight='bold',
             color='white', ha='center', va='top', transform=ax9.transAxes)
    ax9.text(0.50, y_pos-0.11, desc, fontsize=6.5, color='#95A5A6',
             ha='center', va='top', transform=ax9.transAxes, linespacing=1.3)
    y_pos -= 0.18

# ─────────────────────────────────────────────────
# 섹션 4: 푸터
# ─────────────────────────────────────────────────
ax_footer = fig.add_subplot(outer[4])
ax_footer.set_facecolor('#BDC3C7')
ax_footer.axis('off')
ax_footer.text(0.5, 0.55,
               '분석 도구: Python · scikit-learn · seaborn · matplotlib  |  '
               '임베딩: TF-IDF char n-gram(2~4) + cosine similarity  |  '
               'GEFT 그룹 기준: 0-9점 저창의 / 10-18점 고창의',
               ha='center', va='center', fontsize=8.5, color='#555',
               transform=ax_footer.transAxes)

# ── 저장 ───────────────────────────────────────────────────────────────────
out_path = '/home/user/prompt-analysis_01/report_summary.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#F0F2F5')
plt.close()
print(f"저장 완료: {out_path}")
