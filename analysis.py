"""
Prompt Analysis Script
- 1. 한국어 텍스트 임베딩 + 클러스터링
- 2. geft_score와 프롬프트 특성 간 상관관계 분석
- 3. turn별 프롬프트 변화 패턴 분석
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────
import subprocess
subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], capture_output=True)
fm.fontManager.__init__()

nanum = [f.fname for f in fm.fontManager.ttflist if 'Nanum' in f.name]
if nanum:
    plt.rcParams['font.family'] = fm.FontProperties(fname=nanum[0]).get_name()
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("한글 폰트:", plt.rcParams['font.family'])

# ── 데이터 로드 ─────────────────────────────────────────────────────────
df = pd.read_csv('/home/user/prompt-analysis_01/prompt_260316.csv')
df = df[df['prompt_raw'].notna() & (df['prompt_raw'].str.strip() != '')]
df['geft_score'] = pd.to_numeric(df['geft_score'], errors='coerce')
df['turn_num'] = df['turn'].str.extract(r'(\d+)').astype(float)

print(f"데이터: {len(df)}행, 참가자 {df['participant_id'].nunique()}명")
print(f"geft_score 분포:\n{df['geft_score'].describe()}\n")

# ── 텍스트 피처 추출 ────────────────────────────────────────────────────
df['prompt_len'] = df['prompt_raw'].str.len()
df['word_count'] = df['prompt_raw'].str.split().str.len()
df['unique_words'] = df['prompt_raw'].apply(lambda x: len(set(str(x).split())))
df['lexical_diversity'] = df['unique_words'] / df['word_count'].clip(lower=1)
df['question_mark'] = df['prompt_raw'].str.count(r'\?')
df['exclamation'] = df['prompt_raw'].str.count(r'!')
df['num_sentences'] = df['prompt_raw'].str.count(r'[.!?。]') + 1

print("텍스트 피처 추출 완료")

# ════════════════════════════════════════════════════════════════════════
# 1. 한국어 텍스트 임베딩 + 클러스터링 (TF-IDF 기반)
# ════════════════════════════════════════════════════════════════════════
print("\n[1] TF-IDF 임베딩 및 클러스터링 시작...")

from sklearn.feature_extraction.text import TfidfVectorizer

# 음절/자소 단위 분리 (한국어 대응: character n-gram)
tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 4),
    max_features=3000,
    sublinear_tf=True,
)
embeddings_sparse = tfidf.fit_transform(df['prompt_raw'].tolist())
embeddings = embeddings_sparse.toarray()
print(f"TF-IDF 임베딩 shape: {embeddings.shape}")

# 최적 클러스터 수 탐색 (K=2~8)
scaler = StandardScaler()
emb_scaled = scaler.fit_transform(embeddings)

sil_scores = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_scaled)
    sil_scores[k] = silhouette_score(emb_scaled, labels)

best_k = max(sil_scores, key=sil_scores.get)
print(f"Silhouette 점수: {sil_scores}")
print(f"최적 K: {best_k} (실루엣={sil_scores[best_k]:.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(emb_scaled)

# PCA 2D 시각화
pca = PCA(n_components=2, random_state=42)
emb_2d = pca.fit_transform(emb_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 클러스터별 색상
scatter = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1],
                          c=df['cluster'], cmap='tab10', alpha=0.7, s=60)
axes[0].set_title(f'임베딩 클러스터링 (K={best_k}, PCA 2D)')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter, ax=axes[0], label='클러스터')

# geft_score별 색상
sc2 = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1],
                      c=df['geft_score'], cmap='RdYlGn', alpha=0.7, s=60)
axes[1].set_title('임베딩 공간 (geft_score 색상)')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(sc2, ax=axes[1], label='geft_score')

plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig1_embedding_clustering.png', dpi=150)
plt.close()
print("fig1_embedding_clustering.png 저장 완료")

# 실루엣 점수 그래프
fig, ax = plt.subplots(figsize=(7, 4))
ks = list(sil_scores.keys())
ss = list(sil_scores.values())
ax.plot(ks, ss, 'o-', color='steelblue', linewidth=2, markersize=8)
ax.axvline(best_k, color='red', linestyle='--', label=f'최적 K={best_k}')
ax.set_xlabel('클러스터 수 (K)')
ax.set_ylabel('실루엣 점수')
ax.set_title('K별 실루엣 점수')
ax.legend()
plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig1b_silhouette.png', dpi=150)
plt.close()

# 클러스터별 통계
cluster_stats = df.groupby('cluster').agg(
    count=('prompt_id', 'count'),
    mean_geft=('geft_score', 'mean'),
    mean_len=('prompt_len', 'mean'),
    mean_words=('word_count', 'mean')
).round(2)
print("\n클러스터별 통계:")
print(cluster_stats)


# ════════════════════════════════════════════════════════════════════════
# 2. geft_score와 프롬프트 특성 간 상관관계 분석
# ════════════════════════════════════════════════════════════════════════
print("\n[2] geft_score 상관관계 분석...")

# 참가자별 집계 (첫 번째 turn 기준 + 평균)
p_df = df.groupby('participant_id').agg(
    geft_score=('geft_score', 'first'),
    total_turns=('turn_num', 'count'),
    mean_prompt_len=('prompt_len', 'mean'),
    mean_word_count=('word_count', 'mean'),
    mean_lexical_diversity=('lexical_diversity', 'mean'),
    total_questions=('question_mark', 'sum'),
    mean_sentences=('num_sentences', 'mean'),
    first_prompt_len=('prompt_len', 'first'),
    max_prompt_len=('prompt_len', 'max'),
).reset_index()

features = ['total_turns', 'mean_prompt_len', 'mean_word_count',
            'mean_lexical_diversity', 'total_questions',
            'mean_sentences', 'first_prompt_len', 'max_prompt_len']

feature_labels = {
    'total_turns': '총 턴 수',
    'mean_prompt_len': '평균 프롬프트 길이',
    'mean_word_count': '평균 단어 수',
    'mean_lexical_diversity': '어휘 다양성',
    'total_questions': '물음표 사용 수',
    'mean_sentences': '평균 문장 수',
    'first_prompt_len': '첫 프롬프트 길이',
    'max_prompt_len': '최대 프롬프트 길이',
}

corr_results = []
for feat in features:
    valid = p_df[['geft_score', feat]].dropna()
    if len(valid) >= 5:
        r_p, p_p = pearsonr(valid['geft_score'], valid[feat])
        r_s, p_s = spearmanr(valid['geft_score'], valid[feat])
        corr_results.append({
            'feature': feature_labels[feat],
            'pearson_r': round(r_p, 3),
            'pearson_p': round(p_p, 3),
            'spearman_r': round(r_s, 3),
            'spearman_p': round(p_s, 3),
        })

corr_df = pd.DataFrame(corr_results)
print("\n상관관계 분석 결과 (참가자 단위):")
print(corr_df.to_string(index=False))

# 히트맵
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors_p = ['#d32f2f' if p < 0.05 else '#78909c' for p in corr_df['pearson_p']]
bars = axes[0].barh(corr_df['feature'], corr_df['pearson_r'], color=colors_p)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_xlabel('Pearson r')
axes[0].set_title('geft_score × 프롬프트 특성\nPearson 상관계수 (빨강=p<0.05)')
axes[0].set_xlim(-1, 1)

colors_s = ['#d32f2f' if p < 0.05 else '#78909c' for p in corr_df['spearman_p']]
axes[1].barh(corr_df['feature'], corr_df['spearman_r'], color=colors_s)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_xlabel('Spearman ρ')
axes[1].set_title('geft_score × 프롬프트 특성\nSpearman 상관계수 (빨강=p<0.05)')
axes[1].set_xlim(-1, 1)

plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig2_correlation.png', dpi=150)
plt.close()
print("fig2_correlation.png 저장 완료")

# geft_score 그룹 (저=0~9, 고=10~18)
p_df['geft_group'] = pd.cut(p_df['geft_score'],
                             bins=[-1, 9, 18],
                             labels=['저창의(0-9)', '고창의(10-18)'])
group_stats = p_df.groupby('geft_group')[features].mean().round(2)
print("\nGEFT 그룹별 평균:")
print(group_stats)

# 산점도 (주요 특성 4개)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
top_feats = ['total_turns', 'first_prompt_len', 'mean_word_count', 'mean_lexical_diversity']
top_labels = ['총 턴 수', '첫 프롬프트 길이', '평균 단어 수', '어휘 다양성']

for ax, feat, label in zip(axes.flat, top_feats, top_labels):
    valid = p_df[['geft_score', feat, 'participant_id']].dropna()
    ax.scatter(valid['geft_score'], valid[feat], alpha=0.7, s=80, color='steelblue')
    for _, row in valid.iterrows():
        ax.annotate(row['participant_id'], (row['geft_score'], row[feat]),
                    fontsize=6, alpha=0.6)
    m, b = np.polyfit(valid['geft_score'], valid[feat], 1)
    x_line = np.linspace(valid['geft_score'].min(), valid['geft_score'].max(), 100)
    ax.plot(x_line, m * x_line + b, 'r--', linewidth=1.5)
    ax.set_xlabel('geft_score')
    ax.set_ylabel(label)
    ax.set_title(f'geft_score × {label}')

plt.suptitle('geft_score와 프롬프트 특성 산점도', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig2b_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig2b_scatter.png 저장 완료")


# ════════════════════════════════════════════════════════════════════════
# 3. turn별 프롬프트 변화 패턴 분석
# ════════════════════════════════════════════════════════════════════════
print("\n[3] turn별 프롬프트 변화 패턴 분석...")

df_turns = df[df['turn_num'].notna()].copy()

# turn별 평균 프롬프트 길이
turn_agg = df_turns.groupby('turn_num').agg(
    mean_len=('prompt_len', 'mean'),
    mean_words=('word_count', 'mean'),
    mean_lex_div=('lexical_diversity', 'mean'),
    count=('prompt_id', 'count')
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].bar(turn_agg['turn_num'], turn_agg['count'], color='steelblue', alpha=0.8)
axes[0].set_xlabel('Turn 번호')
axes[0].set_ylabel('프롬프트 수')
axes[0].set_title('Turn별 프롬프트 수')
axes[0].set_xticks(turn_agg['turn_num'])

axes[1].plot(turn_agg['turn_num'], turn_agg['mean_len'], 'o-', color='tomato',
             linewidth=2, markersize=8, label='평균 길이')
axes[1].fill_between(turn_agg['turn_num'], turn_agg['mean_len'], alpha=0.2, color='tomato')
axes[1].set_xlabel('Turn 번호')
axes[1].set_ylabel('평균 글자 수')
axes[1].set_title('Turn별 평균 프롬프트 길이')
axes[1].set_xticks(turn_agg['turn_num'])
ax2 = axes[1].twinx()
ax2.plot(turn_agg['turn_num'], turn_agg['mean_words'], 's--', color='royalblue',
         linewidth=1.5, markersize=6, label='단어 수')
ax2.set_ylabel('평균 단어 수', color='royalblue')

axes[2].plot(turn_agg['turn_num'], turn_agg['mean_lex_div'], 'D-', color='seagreen',
             linewidth=2, markersize=8)
axes[2].fill_between(turn_agg['turn_num'], turn_agg['mean_lex_div'], alpha=0.2, color='seagreen')
axes[2].set_xlabel('Turn 번호')
axes[2].set_ylabel('어휘 다양성 (type-token ratio)')
axes[2].set_title('Turn별 평균 어휘 다양성')
axes[2].set_xticks(turn_agg['turn_num'])

plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig3_turn_pattern.png', dpi=150)
plt.close()
print("fig3_turn_pattern.png 저장 완료")

# 개인별 turn 프로파일 (geft 그룹별)
df_turns2 = df_turns.copy()
df_turns2['geft_group'] = pd.cut(df_turns2['geft_score'],
                                  bins=[-1, 9, 18],
                                  labels=['저창의(0-9)', '고창의(10-18)'])

group_turn = df_turns2.groupby(['geft_group', 'turn_num']).agg(
    mean_len=('prompt_len', 'mean'),
    mean_words=('word_count', 'mean'),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
palette = {'저창의(0-9)': '#e74c3c', '고창의(10-18)': '#2ecc71'}

for group, gdf in group_turn.groupby('geft_group'):
    axes[0].plot(gdf['turn_num'], gdf['mean_len'], 'o-',
                 label=str(group), color=palette.get(str(group), 'gray'),
                 linewidth=2, markersize=7)
    axes[1].plot(gdf['turn_num'], gdf['mean_words'], 's-',
                 label=str(group), color=palette.get(str(group), 'gray'),
                 linewidth=2, markersize=7)

axes[0].set_xlabel('Turn 번호')
axes[0].set_ylabel('평균 프롬프트 길이 (글자)')
axes[0].set_title('GEFT 그룹별 Turn에 따른 프롬프트 길이 변화')
axes[0].legend()

axes[1].set_xlabel('Turn 번호')
axes[1].set_ylabel('평균 단어 수')
axes[1].set_title('GEFT 그룹별 Turn에 따른 단어 수 변화')
axes[1].legend()

plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig3b_turn_by_geft.png', dpi=150)
plt.close()
print("fig3b_turn_by_geft.png 저장 완료")

# 임베딩 기반 turn 간 코사인 유사도 (동일 참가자 연속 턴, TF-IDF)
from sklearn.metrics.pairwise import cosine_similarity

cos_sim_rows = []
for pid, pdata in df_turns.groupby('participant_id'):
    pdata = pdata.sort_values('turn_num')
    if len(pdata) < 2:
        continue
    p_embs = tfidf.transform(pdata['prompt_raw'].tolist()).toarray()
    for i in range(len(pdata) - 1):
        sim = cosine_similarity(p_embs[i:i+1], p_embs[i+1:i+2])[0][0]
        cos_sim_rows.append({
            'participant_id': pid,
            'from_turn': pdata.iloc[i]['turn_num'],
            'to_turn': pdata.iloc[i+1]['turn_num'],
            'cosine_sim': sim,
            'geft_score': pdata.iloc[i]['geft_score'],
        })

cos_df = pd.DataFrame(cos_sim_rows)
cos_df['geft_group'] = pd.cut(cos_df['geft_score'],
                               bins=[-1, 9, 18],
                               labels=['저창의(0-9)', '고창의(10-18)'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Turn 전환 단계별 평균 유사도
step_sim = cos_df.groupby('from_turn')['cosine_sim'].mean()
axes[0].bar(step_sim.index, step_sim.values, color='mediumpurple', alpha=0.8)
axes[0].set_xlabel('Turn 번호 (→다음 Turn)')
axes[0].set_ylabel('평균 코사인 유사도')
axes[0].set_title('연속 Turn 간 의미적 유사도 변화')
axes[0].set_ylim(0, 1)
for i, (x, y) in enumerate(zip(step_sim.index, step_sim.values)):
    axes[0].text(x, y + 0.01, f'{y:.2f}', ha='center', fontsize=8)

# GEFT 그룹별 평균 유사도 분포
valid_groups = cos_df['geft_group'].dropna().unique()
for group in valid_groups:
    gdf = cos_df[cos_df['geft_group'] == group]['cosine_sim']
    axes[1].hist(gdf, bins=15, alpha=0.6, label=str(group),
                 color=palette.get(str(group), 'gray'))
axes[1].set_xlabel('코사인 유사도')
axes[1].set_ylabel('빈도')
axes[1].set_title('GEFT 그룹별 연속 Turn 유사도 분포')
axes[1].legend()

plt.tight_layout()
plt.savefig('/home/user/prompt-analysis_01/fig3c_turn_similarity.png', dpi=150)
plt.close()
print("fig3c_turn_similarity.png 저장 완료")

# ── 요약 출력 ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("분석 완료 요약")
print("="*60)
print(f"\n[1] 클러스터링: 최적 K={best_k}, 실루엣={sil_scores[best_k]:.3f}")
print(cluster_stats.to_string())
print(f"\n[2] 상관관계 (유의미한 항목, p<0.1):")
sig = corr_df[(corr_df['pearson_p'] < 0.1) | (corr_df['spearman_p'] < 0.1)]
print(sig[['feature', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p']].to_string(index=False))
print(f"\n[3] turn별 패턴:")
print(f"  - turn_01 평균 길이: {turn_agg[turn_agg['turn_num']==1]['mean_len'].values[0]:.0f}자")
print(f"  - turn 전체 평균 코사인 유사도: {cos_df['cosine_sim'].mean():.3f}")
geft_sim = cos_df.groupby('geft_group')['cosine_sim'].mean()
print(f"  - GEFT 그룹별 평균 유사도:\n{geft_sim}")
print("\n저장된 파일:")
for f in ['fig1_embedding_clustering.png', 'fig1b_silhouette.png',
          'fig2_correlation.png', 'fig2b_scatter.png',
          'fig3_turn_pattern.png', 'fig3b_turn_by_geft.png',
          'fig3c_turn_similarity.png']:
    print(f"  - {f}")
