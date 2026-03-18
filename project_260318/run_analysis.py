"""
프롬프트 임베딩 분석 파이프라인
================================
1) 클러스터링 / UMAP 시각화
2) 턴별 프롬프트 변화 추적 (FI vs FD)
3) GEFT 점수와 프롬프트 특성 상관관계
4) 맥락 중심 vs 객체 중심 키워드 분석

입력: output/prompt_preprocessed.csv (또는 data/prompt_250318_captioned.csv)
출력: output/ 에 PNG 그래프 + CSV 요약
"""

import os
import re
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats as sp_stats

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import umap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 폰트 설정
# ─────────────────────────────────────────────
def setup_font():
    """한글 폰트 설정 (Noto Sans KR 또는 시스템 폰트)"""
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJKkr-Regular.otf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    # Also search broadly
    import glob
    candidates += glob.glob("/usr/share/fonts/**/Noto*CJK*.ttc", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/Noto*CJK*.otf", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/Nanum*.ttf", recursive=True)

    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            print(f"✅ 한글 폰트 설정: {path}")
            return True

    # Fallback: try to install
    try:
        os.system("apt-get install -y fonts-noto-cjk > /dev/null 2>&1")
        for path in glob.glob("/usr/share/fonts/**/Noto*CJK*.ttc", recursive=True):
            if os.path.exists(path):
                fm.fontManager.addfont(path)
                prop = fm.FontProperties(fname=path)
                plt.rcParams["font.family"] = prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                print(f"✅ 한글 폰트 설치 및 설정: {path}")
                return True
    except Exception:
        pass

    print("⚠️ 한글 폰트를 찾을 수 없습니다. 영문으로 레이블을 표시합니다.")
    return False


HAS_KOREAN_FONT = setup_font()


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """전처리된 CSV 로드"""
    csv_path = OUTPUT_DIR / "prompt_preprocessed.csv"
    if not csv_path.exists():
        csv_path = BASE_DIR / "data" / "prompt_250318_captioned.csv"
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["geft_score"] = pd.to_numeric(df["geft_score"], errors="coerce")
    df["char_count"] = pd.to_numeric(df["char_count"], errors="coerce")
    df["char_count_no_space"] = pd.to_numeric(df["char_count_no_space"], errors="coerce")
    df["turn_num"] = df["turn"].str.extract(r"(\d+)").astype(int)
    print(f"✅ 데이터 로드: {len(df)}행, {df['participant_id'].nunique()}명 참가자")
    return df


def build_embeddings(df):
    """TF-IDF 기반 임베딩 생성 + 저장"""
    texts = df["prompt_combined"].fillna(df["prompt_raw"].fillna("")).tolist()
    texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]

    vectorizer = TfidfVectorizer(max_features=500, analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(texts)
    embeddings = tfidf_matrix.toarray()

    # 저장
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    meta = {
        "method": "tfidf_char_wb_ngram_2_4",
        "max_features": 500,
        "n_samples": embeddings.shape[0],
        "n_features": embeddings.shape[1],
        "prompt_ids": df["prompt_id"].tolist(),
        "participant_ids": df["participant_id"].tolist(),
    }
    import json
    with open(OUTPUT_DIR / "embedding_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ TF-IDF 임베딩 생성: {embeddings.shape}")
    return embeddings


# ═══════════════════════════════════════════════
# 분석 1: 클러스터링 / UMAP 시각화
# ═══════════════════════════════════════════════
def analysis_1_clustering_umap(df, embeddings):
    print("\n" + "=" * 60)
    print("📊 분석 1: 클러스터링 / UMAP 시각화")
    print("=" * 60)

    # --- 최적 클러스터 수 탐색 ---
    sil_scores = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels)
        sil_scores.append(sil)
        print(f"  k={k}: silhouette={sil:.3f}")

    best_k = list(K_range)[np.argmax(sil_scores)]
    print(f"  → 최적 k = {best_k}")

    # --- K-Means 클러스터링 ---
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(embeddings)

    # --- UMAP 2D ---
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_2d = reducer.fit_transform(embeddings)
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]

    # --- 시각화: 그룹(FI/FD) 기반 UMAP ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 왼쪽: FI/FD 그룹
    colors_group = {"FI": "#2196F3", "FD": "#FF5722"}
    for grp, color in colors_group.items():
        mask = df["participant_group"] == grp
        axes[0].scatter(
            df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
            c=color, label=grp, alpha=0.7, s=60, edgecolors="white", linewidths=0.5,
        )
    axes[0].set_title("UMAP by Cognitive Style (FI vs FD)", fontsize=14)
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].legend(title="Group", fontsize=11)

    # 오른쪽: 클러스터
    palette = sns.color_palette("Set2", best_k)
    for c in range(best_k):
        mask = df["cluster"] == c
        axes[1].scatter(
            df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
            c=[palette[c]], label=f"Cluster {c}", alpha=0.7, s=60,
            edgecolors="white", linewidths=0.5,
        )
    axes[1].set_title(f"UMAP by K-Means Cluster (k={best_k})", fontsize=14)
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].legend(title="Cluster", fontsize=11)

    plt.tight_layout()
    path = OUTPUT_DIR / "analysis1_umap_clustering.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 저장: {path}")

    # --- 클러스터별 그룹 분포 CSV ---
    ct = pd.crosstab(df["cluster"], df["participant_group"], margins=True)
    ct_path = OUTPUT_DIR / "analysis1_cluster_group_distribution.csv"
    ct.to_csv(ct_path, encoding="utf-8-sig")
    print(f"  💾 저장: {ct_path}")

    return df


# ═══════════════════════════════════════════════
# 분석 2: 턴별 프롬프트 변화 추적 (FI vs FD)
# ═══════════════════════════════════════════════
def analysis_2_turn_tracking(df, embeddings):
    print("\n" + "=" * 60)
    print("📊 분석 2: 턴별 프롬프트 변화 추적 (FI vs FD)")
    print("=" * 60)

    # --- 턴별 평균 글자 수 (FI vs FD) ---
    turn_stats = (
        df.groupby(["participant_group", "turn_num"])
        .agg(
            mean_char=("char_count", "mean"),
            std_char=("char_count", "std"),
            count=("char_count", "size"),
            mean_char_no_space=("char_count_no_space", "mean"),
        )
        .reset_index()
    )
    turn_stats["std_char"] = turn_stats["std_char"].fillna(0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) 턴별 평균 글자 수
    for grp, color in [("FI", "#2196F3"), ("FD", "#FF5722")]:
        sub = turn_stats[turn_stats["participant_group"] == grp]
        axes[0, 0].plot(sub["turn_num"], sub["mean_char"], "o-", color=color, label=grp, linewidth=2)
        axes[0, 0].fill_between(
            sub["turn_num"],
            sub["mean_char"] - sub["std_char"],
            sub["mean_char"] + sub["std_char"],
            alpha=0.15, color=color,
        )
    axes[0, 0].set_title("Mean Character Count per Turn", fontsize=13)
    axes[0, 0].set_xlabel("Turn")
    axes[0, 0].set_ylabel("Character Count")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # (b) 턴별 프롬프트 수 (FI vs FD)
    for grp, color in [("FI", "#2196F3"), ("FD", "#FF5722")]:
        sub = turn_stats[turn_stats["participant_group"] == grp]
        axes[0, 1].bar(
            sub["turn_num"] + (0.2 if grp == "FI" else -0.2),
            sub["count"], width=0.4, color=color, label=grp, alpha=0.8,
        )
    axes[0, 1].set_title("Prompt Count per Turn", fontsize=13)
    axes[0, 1].set_xlabel("Turn")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # (c) 턴간 임베딩 코사인 거리 (프롬프트 변화량)
    from sklearn.metrics.pairwise import cosine_distances

    change_records = []
    for pid, grp_df in df.groupby("participant_id"):
        grp_df = grp_df.sort_values("turn_num")
        indices = grp_df.index.tolist()
        group = grp_df["participant_group"].iloc[0]
        for i in range(1, len(indices)):
            prev_idx = indices[i - 1]
            curr_idx = indices[i]
            dist = cosine_distances(
                embeddings[prev_idx].reshape(1, -1),
                embeddings[curr_idx].reshape(1, -1),
            )[0, 0]
            change_records.append({
                "participant_id": pid,
                "participant_group": group,
                "from_turn": grp_df.loc[prev_idx, "turn_num"],
                "to_turn": grp_df.loc[curr_idx, "turn_num"],
                "cosine_distance": dist,
            })

    change_df = pd.DataFrame(change_records)
    change_agg = (
        change_df.groupby(["participant_group", "to_turn"])["cosine_distance"]
        .agg(["mean", "std"])
        .reset_index()
    )
    change_agg["std"] = change_agg["std"].fillna(0)

    for grp, color in [("FI", "#2196F3"), ("FD", "#FF5722")]:
        sub = change_agg[change_agg["participant_group"] == grp]
        axes[1, 0].plot(sub["to_turn"], sub["mean"], "o-", color=color, label=grp, linewidth=2)
        axes[1, 0].fill_between(
            sub["to_turn"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
            alpha=0.15, color=color,
        )
    axes[1, 0].set_title("Prompt Change (Cosine Distance) per Turn", fontsize=13)
    axes[1, 0].set_xlabel("Turn")
    axes[1, 0].set_ylabel("Cosine Distance from Previous Turn")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # (d) prompt_type 분포 (FI vs FD)
    type_dist = pd.crosstab(df["participant_group"], df["prompt_type"], normalize="index") * 100
    type_dist.plot(kind="bar", ax=axes[1, 1], rot=0, colormap="Set2")
    axes[1, 1].set_title("Prompt Type Distribution (%)", fontsize=13)
    axes[1, 1].set_xlabel("Group")
    axes[1, 1].set_ylabel("Percentage (%)")
    axes[1, 1].legend(title="Type")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "analysis2_turn_tracking.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 저장: {path}")

    # CSV 저장
    turn_stats.to_csv(OUTPUT_DIR / "analysis2_turn_stats.csv", index=False, encoding="utf-8-sig")
    change_df.to_csv(OUTPUT_DIR / "analysis2_turn_change.csv", index=False, encoding="utf-8-sig")
    print(f"  💾 저장: analysis2_turn_stats.csv, analysis2_turn_change.csv")


# ═══════════════════════════════════════════════
# 분석 3: GEFT 점수와 프롬프트 특성 상관관계
# ═══════════════════════════════════════════════
def analysis_3_geft_correlation(df):
    print("\n" + "=" * 60)
    print("📊 분석 3: GEFT 점수와 프롬프트 특성 상관관계")
    print("=" * 60)

    # 참가자별 집계
    participant_agg = (
        df.groupby(["participant_id", "participant_group", "geft_score"])
        .agg(
            total_turns=("turn_num", "max"),
            mean_char=("char_count", "mean"),
            total_char=("char_count", "sum"),
            mean_char_no_space=("char_count_no_space", "mean"),
            n_prompts=("prompt_id", "count"),
            n_refinement=("prompt_type", lambda x: (x == "refinement").sum()),
            n_selection=("prompt_type", lambda x: (x == "selection").sum()),
            n_initial=("prompt_type", lambda x: (x == "initial").sum()),
            n_reference_img=("img_type", lambda x: (x == "reference").sum()),
        )
        .reset_index()
    )
    participant_agg["refinement_ratio"] = participant_agg["n_refinement"] / participant_agg["n_prompts"]

    # 상관계수 계산
    corr_vars = ["total_turns", "mean_char", "total_char", "n_prompts",
                 "refinement_ratio", "n_reference_img"]
    corr_labels = ["Total Turns", "Mean Char Count", "Total Char Count",
                   "N Prompts", "Refinement Ratio", "N Ref Images"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    corr_results = []
    for i, (var, label) in enumerate(zip(corr_vars, corr_labels)):
        ax = axes_flat[i]
        x = participant_agg["geft_score"]
        y = participant_agg[var]

        r, p = sp_stats.pearsonr(x, y)
        corr_results.append({"variable": var, "label": label, "pearson_r": round(r, 3), "p_value": round(p, 4)})

        # 그룹별 색상
        for grp, color, marker in [("FI", "#2196F3", "o"), ("FD", "#FF5722", "s")]:
            mask = participant_agg["participant_group"] == grp
            ax.scatter(x[mask], y[mask], c=color, marker=marker, s=80, alpha=0.7,
                       label=grp, edgecolors="white", linewidths=0.5)

        # 회귀선
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_range, p_line(x_range), "--", color="gray", alpha=0.6)

        ax.set_title(f"{label}\nr={r:.3f}, p={p:.3f}", fontsize=11)
        ax.set_xlabel("GEFT Score")
        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("GEFT Score vs Prompt Characteristics", fontsize=15, y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / "analysis3_geft_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 저장: {path}")

    # 상관관계 결과 CSV
    corr_df = pd.DataFrame(corr_results)
    corr_df.to_csv(OUTPUT_DIR / "analysis3_geft_correlation.csv", index=False, encoding="utf-8-sig")
    participant_agg.to_csv(OUTPUT_DIR / "analysis3_participant_summary.csv", index=False, encoding="utf-8-sig")
    print(f"  💾 저장: analysis3_geft_correlation.csv, analysis3_participant_summary.csv")

    # 통계 출력
    for row in corr_results:
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"  {row['label']:25s}: r={row['pearson_r']:+.3f}, p={row['p_value']:.4f} {sig}")


# ═══════════════════════════════════════════════
# 분석 4: 맥락 중심 vs 객체 중심 키워드 분석
# ═══════════════════════════════════════════════
def analysis_4_keyword_analysis(df):
    print("\n" + "=" * 60)
    print("📊 분석 4: 맥락 중심 vs 객체 중심 키워드 분석")
    print("=" * 60)

    # 키워드 사전 정의
    context_keywords = {
        # 공간/환경/배경 관련
        "분위기": "분위기", "환경": "환경", "공간": "공간", "배경": "배경",
        "인테리어": "인테리어", "감성": "감성", "느낌": "느낌", "스타일": "스타일",
        "장소": "장소", "실내": "실내", "야외": "야외", "현실": "현실",
        "상황": "상황", "사용": "사용", "편의": "편의", "편리": "편리",
        "불편": "불편", "경험": "경험", "사용자": "사용자",
        # 감정/추상 표현
        "깔끔": "깔끔", "심미": "심미", "자연": "자연", "모던": "모던",
        "트렌디": "트렌디", "미니멀": "미니멀", "따뜻": "따뜻",
        "학교": "학교", "강의실": "강의실", "건물": "건물", "캠퍼스": "캠퍼스",
    }

    object_keywords = {
        # 물리적 객체/부품/형태
        "디자인": "디자인", "색상": "색상", "색깔": "색깔", "크기": "크기",
        "모양": "모양", "형태": "형태", "소재": "소재", "재질": "재질",
        "버튼": "버튼", "화면": "화면", "기능": "기능", "구조": "구조",
        "제품": "제품", "기기": "기기", "디바이스": "디바이스",
        "충전": "충전", "무선": "무선", "레일": "레일", "멀티탭": "멀티탭",
        "천장": "천장", "줄": "줄", "패드": "패드", "키오스크": "키오스크",
        "표지판": "표지판", "지도": "지도", "화살표": "화살표",
        "USB": "USB", "콘센트": "콘센트", "발열": "발열",
        "가로": "가로", "세로": "세로", "곡선": "곡선",
    }

    def count_keywords(text, kw_dict):
        if not isinstance(text, str):
            return 0, {}
        total = 0
        found = {}
        for kw in kw_dict:
            cnt = text.count(kw)
            if cnt > 0:
                total += cnt
                found[kw] = cnt
        return total, found

    # 각 행에 대해 키워드 카운트
    context_counts = []
    object_counts = []
    context_details_all = []
    object_details_all = []

    for _, row in df.iterrows():
        text = str(row.get("prompt_raw", "")) + " " + str(row.get("prompt_combined", ""))
        c_cnt, c_detail = count_keywords(text, context_keywords)
        o_cnt, o_detail = count_keywords(text, object_keywords)
        context_counts.append(c_cnt)
        object_counts.append(o_cnt)
        context_details_all.append(c_detail)
        object_details_all.append(o_detail)

    df["context_kw_count"] = context_counts
    df["object_kw_count"] = object_counts
    df["kw_total"] = df["context_kw_count"] + df["object_kw_count"]
    df["context_ratio"] = df["context_kw_count"] / df["kw_total"].replace(0, np.nan)
    df["object_ratio"] = df["object_kw_count"] / df["kw_total"].replace(0, np.nan)

    # --- 그룹별 집계 ---
    group_kw = (
        df.groupby("participant_group")
        .agg(
            mean_context=("context_kw_count", "mean"),
            mean_object=("object_kw_count", "mean"),
            mean_context_ratio=("context_ratio", "mean"),
            mean_object_ratio=("object_ratio", "mean"),
        )
        .reset_index()
    )
    print("\n  그룹별 키워드 평균:")
    print(group_kw.to_string(index=False))

    # 그룹별 전체 키워드 빈도 집계
    fi_context_all = Counter()
    fi_object_all = Counter()
    fd_context_all = Counter()
    fd_object_all = Counter()

    for idx, row in df.iterrows():
        grp = row["participant_group"]
        c_detail = context_details_all[idx]
        o_detail = object_details_all[idx]
        if grp == "FI":
            fi_context_all.update(c_detail)
            fi_object_all.update(o_detail)
        else:
            fd_context_all.update(c_detail)
            fd_object_all.update(o_detail)

    # --- 시각화 ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # (a) FI vs FD: 맥락 vs 객체 키워드 비율
    x_labels = ["FI", "FD"]
    ctx_means = [group_kw.loc[group_kw["participant_group"] == g, "mean_context"].values[0] for g in x_labels]
    obj_means = [group_kw.loc[group_kw["participant_group"] == g, "mean_object"].values[0] for g in x_labels]

    x_pos = np.arange(len(x_labels))
    w = 0.35
    axes[0, 0].bar(x_pos - w / 2, ctx_means, w, label="Context Keywords", color="#4CAF50", alpha=0.8)
    axes[0, 0].bar(x_pos + w / 2, obj_means, w, label="Object Keywords", color="#9C27B0", alpha=0.8)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(x_labels)
    axes[0, 0].set_title("Mean Keyword Count: Context vs Object", fontsize=13)
    axes[0, 0].set_ylabel("Mean Count per Prompt")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis="y")

    # (b) 턴별 키워드 비율 변화
    turn_kw = (
        df.groupby(["participant_group", "turn_num"])
        .agg(mean_ctx=("context_kw_count", "mean"), mean_obj=("object_kw_count", "mean"))
        .reset_index()
    )
    for grp, color_c, color_o, ls in [("FI", "#2196F3", "#03A9F4", "-"), ("FD", "#FF5722", "#FF9800", "--")]:
        sub = turn_kw[turn_kw["participant_group"] == grp]
        axes[0, 1].plot(sub["turn_num"], sub["mean_ctx"], f"o{ls}", color=color_c, label=f"{grp} Context", linewidth=2)
        axes[0, 1].plot(sub["turn_num"], sub["mean_obj"], f"s{ls}", color=color_o, label=f"{grp} Object", linewidth=1.5, alpha=0.7)
    axes[0, 1].set_title("Keyword Trend by Turn", fontsize=13)
    axes[0, 1].set_xlabel("Turn")
    axes[0, 1].set_ylabel("Mean Keyword Count")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # (c) FI 상위 키워드
    fi_all = fi_context_all + fi_object_all
    fi_top = fi_all.most_common(15)
    if fi_top:
        kws, cnts = zip(*fi_top)
        colors_c = ["#4CAF50" if k in context_keywords else "#9C27B0" for k in kws]
        axes[1, 0].barh(range(len(kws)), cnts, color=colors_c, alpha=0.8)
        axes[1, 0].set_yticks(range(len(kws)))
        axes[1, 0].set_yticklabels(kws)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_title("FI Group: Top 15 Keywords", fontsize=13)
        axes[1, 0].set_xlabel("Frequency")
        legend_elements = [
            Line2D([0], [0], color="#4CAF50", lw=6, label="Context"),
            Line2D([0], [0], color="#9C27B0", lw=6, label="Object"),
        ]
        axes[1, 0].legend(handles=legend_elements, fontsize=9)
        axes[1, 0].grid(alpha=0.3, axis="x")

    # (d) FD 상위 키워드
    fd_all = fd_context_all + fd_object_all
    fd_top = fd_all.most_common(15)
    if fd_top:
        kws, cnts = zip(*fd_top)
        colors_c = ["#4CAF50" if k in context_keywords else "#9C27B0" for k in kws]
        axes[1, 1].barh(range(len(kws)), cnts, color=colors_c, alpha=0.8)
        axes[1, 1].set_yticks(range(len(kws)))
        axes[1, 1].set_yticklabels(kws)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_title("FD Group: Top 15 Keywords", fontsize=13)
        axes[1, 1].set_xlabel("Frequency")
        axes[1, 1].legend(handles=legend_elements, fontsize=9)
        axes[1, 1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    path = OUTPUT_DIR / "analysis4_keyword_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 저장: {path}")

    # 키워드 빈도 CSV
    kw_records = []
    for kw, cnt in fi_context_all.most_common():
        kw_records.append({"group": "FI", "type": "context", "keyword": kw, "count": cnt})
    for kw, cnt in fi_object_all.most_common():
        kw_records.append({"group": "FI", "type": "object", "keyword": kw, "count": cnt})
    for kw, cnt in fd_context_all.most_common():
        kw_records.append({"group": "FD", "type": "context", "keyword": kw, "count": cnt})
    for kw, cnt in fd_object_all.most_common():
        kw_records.append({"group": "FD", "type": "object", "keyword": kw, "count": cnt})

    kw_df = pd.DataFrame(kw_records)
    kw_df.to_csv(OUTPUT_DIR / "analysis4_keyword_frequency.csv", index=False, encoding="utf-8-sig")

    # 행별 키워드 카운트 CSV
    kw_per_row = df[["prompt_id", "participant_id", "participant_group", "turn_num",
                      "context_kw_count", "object_kw_count", "context_ratio", "object_ratio"]].copy()
    kw_per_row.to_csv(OUTPUT_DIR / "analysis4_keyword_per_prompt.csv", index=False, encoding="utf-8-sig")
    print(f"  💾 저장: analysis4_keyword_frequency.csv, analysis4_keyword_per_prompt.csv")

    # 통계 검정 (Mann-Whitney)
    fi_ctx = df[df["participant_group"] == "FI"]["context_kw_count"]
    fd_ctx = df[df["participant_group"] == "FD"]["context_kw_count"]
    fi_obj = df[df["participant_group"] == "FI"]["object_kw_count"]
    fd_obj = df[df["participant_group"] == "FD"]["object_kw_count"]

    u1, p1 = sp_stats.mannwhitneyu(fi_ctx, fd_ctx, alternative="two-sided")
    u2, p2 = sp_stats.mannwhitneyu(fi_obj, fd_obj, alternative="two-sided")
    print(f"\n  Mann-Whitney U 검정:")
    print(f"    Context Keywords (FI vs FD): U={u1:.1f}, p={p1:.4f}")
    print(f"    Object Keywords  (FI vs FD): U={u2:.1f}, p={p2:.4f}")


# ═══════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════
def main():
    print("🚀 프롬프트 분석 파이프라인 시작\n")

    df = load_data()
    embeddings = build_embeddings(df)

    df = analysis_1_clustering_umap(df, embeddings)
    analysis_2_turn_tracking(df, embeddings)
    analysis_3_geft_correlation(df)
    analysis_4_keyword_analysis(df)

    print("\n" + "=" * 60)
    print("🎉 모든 분석 완료!")
    print(f"   출력 디렉토리: {OUTPUT_DIR}")
    print("=" * 60)

    # 최종 파일 목록
    for f in sorted(OUTPUT_DIR.glob("analysis*")):
        size = f.stat().st_size
        print(f"   {f.name:50s} ({size:,} bytes)")


if __name__ == "__main__":
    main()
