"""
프롬프트 텍스트 임베딩 파이프라인
=================================
1단계: 전처리 (파생 컬럼 생성)
2단계: 임베딩 (OpenAI text-embedding-3-large)
3단계: 결과 저장

사용법 (클로드 코드에서):
    python embedding_pipeline.py --api-key YOUR_OPENAI_API_KEY

옵션:
    --input        입력 CSV 경로 (기본: data/prompt_250318.csv)
    --output-dir   출력 디렉토리 (기본: output/)
    --model        임베딩 모델 (기본: text-embedding-3-large)
    --skip-embedding  전처리만 실행하고 임베딩은 건너뛰기
"""

import os
import csv
import json
import argparse
import time
from pathlib import Path

# ============================================================
# 1단계: 전처리
# ============================================================

def load_csv(filepath):
    """CSV 파일 로드"""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"✅ 데이터 로드 완료: {len(rows)}행")
    return rows


def add_char_counts(rows):
    """글자 수 컬럼 추가 (공백 포함/제외)"""
    for r in rows:
        text = r.get("prompt_raw", "")
        r["char_count"] = len(text)
        r["char_count_no_space"] = len(text.replace(" ", ""))
    print("✅ char_count, char_count_no_space 생성 완료")


def add_prompt_cumulative(rows):
    """참여자별 턴 누적 프롬프트 생성"""
    # 참여자별로 그룹화
    from collections import defaultdict
    participant_rows = defaultdict(list)
    for r in rows:
        participant_rows[r["participant_id"]].append(r)

    for pid, p_rows in participant_rows.items():
        # 턴 번호 기준 정렬
        p_rows.sort(key=lambda x: int(x["turn"].replace("turn_", "")))
        cumulative = []
        for r in p_rows:
            text = r.get("prompt_raw", "").strip()
            if text:
                cumulative.append(text)
            r["prompt_cumulative"] = " ".join(cumulative)

    print("✅ prompt_cumulative 생성 완료")


def resolve_selection_text(row):
    """selection 타입일 때 선택지 번호를 실제 텍스트로 치환"""
    if row.get("prompt_type") != "selection":
        return row.get("prompt_raw", "")

    raw = row.get("prompt_raw", "").strip()
    context = row.get("prompt_context", "")

    if not context:
        return raw

    # 선택지 파싱: "1️⃣ ...\n2️⃣ ...\n3️⃣ ..." 또는 "1. ...\n2. ..." 형태
    import re
    options = {}
    # 이모지 번호 패턴 (1️⃣, 2️⃣, 3️⃣)
    emoji_pattern = re.findall(r"(\d)️⃣\s*(.+?)(?=\d️⃣|$)", context, re.DOTALL)
    if emoji_pattern:
        for num, text in emoji_pattern:
            options[num] = text.strip()
    else:
        # 일반 번호 패턴 (1. ..., 2. ...)
        num_pattern = re.findall(r"(\d+)[.)]\s*(.+?)(?=\d+[.)]|$)", context, re.DOTALL)
        for num, text in num_pattern:
            options[num] = text.strip()

    # raw가 숫자만 포함된 경우 (예: "3", "1번") → 선택지 치환
    selected = re.search(r"(\d)", raw)
    if selected and selected.group(1) in options:
        # raw가 짧은 선택 응답인지 확인 (10자 이하)
        if len(raw) <= 10:
            return options[selected.group(1)]

    # raw가 긍정 응답만 있는 경우 (예: "엉", "응", "네") → context 전체 요약으로 대체
    affirmative = {"엉", "응", "네", "ㅇㅇ", "좋아", "그래", "ok", "ㅇ"}
    if raw.strip().lower() in affirmative and context:
        # 첫 번째 선택지 또는 context 자체를 반환
        return context.strip()

    # 그 외 (자발적 텍스트가 충분한 경우) → 원문 반환
    return raw


def add_prompt_combined(rows):
    """prompt_raw + image_caption_long 합친 텍스트 생성
    - selection 타입: 선택지 텍스트로 치환
    - 이미지 캡션: [참고이미지] 태그 추가 (캡션이 있는 경우)
    """
    for r in rows:
        parts = []

        # 텍스트 부분
        if r.get("prompt_type") == "selection":
            resolved = resolve_selection_text(r)
            if resolved:
                parts.append(f"[프롬프트] {resolved}")
        else:
            raw = r.get("prompt_raw", "").strip()
            if raw:
                parts.append(f"[프롬프트] {raw}")

        # 이미지 캡션 부분 (있는 경우)
        caption = r.get("image_caption_long", "").strip()
        if caption:
            parts.append(f"[참고이미지] {caption}")

        r["prompt_combined"] = " ".join(parts) if parts else ""

    print("✅ prompt_combined 생성 완료")


def preprocess(rows):
    """전체 전처리 실행"""
    print("\n📋 전처리 시작...")
    add_char_counts(rows)
    add_prompt_cumulative(rows)
    add_prompt_combined(rows)
    print("📋 전처리 완료!\n")
    return rows


# ============================================================
# 2단계: 임베딩
# ============================================================

def get_embeddings_batch(texts, api_key, model="text-embedding-3-large", batch_size=50):
    """OpenAI API로 텍스트 임베딩 생성 (배치 처리)"""
    import urllib.request
    import urllib.error

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        # 빈 텍스트 처리
        batch = [t if t.strip() else " " for t in batch]

        payload = json.dumps({
            "model": model,
            "input": batch,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
                print(f"  배치 {batch_num}/{total_batches} 완료 ({len(batch)}개)")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            print(f"  ❌ API 오류 (배치 {batch_num}): {e.code} - {error_body}")
            # 빈 벡터로 채우기
            all_embeddings.extend([[] for _ in batch])

        # Rate limit 방지
        if batch_num < total_batches:
            time.sleep(0.5)

    return all_embeddings


def run_embedding(rows, api_key, model="text-embedding-3-large"):
    """임베딩 실행"""
    print("🔢 임베딩 시작...")

    # prompt_combined를 임베딩 입력으로 사용
    texts = [r.get("prompt_combined", r.get("prompt_raw", "")) for r in rows]

    print(f"  총 {len(texts)}개 텍스트 임베딩 중...")
    embeddings = get_embeddings_batch(texts, api_key, model)

    # 결과를 rows에 추가
    for r, emb in zip(rows, embeddings):
        r["embedding"] = emb

    dim = len(embeddings[0]) if embeddings and embeddings[0] else 0
    print(f"🔢 임베딩 완료! (차원: {dim})\n")
    return rows


# ============================================================
# 3단계: 저장
# ============================================================

def save_results(rows, output_dir):
    """결과 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 전처리된 CSV 저장 (임베딩 벡터 제외)
    csv_path = output_path / "prompt_preprocessed.csv"
    fieldnames = [
        "prompt_id", "turn", "participant_id", "geft_score",
        "participant_group", "prompt_type", "prompt_context",
        "prompt_raw", "img_type", "prompt_img",
        "char_count", "char_count_no_space",
        "prompt_combined", "prompt_cumulative",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ 전처리 CSV 저장: {csv_path}")

    # 2. 임베딩 벡터 저장 (JSON)
    has_embeddings = any(r.get("embedding") for r in rows)
    if has_embeddings:
        embeddings_data = {
            "model": "text-embedding-3-large",
            "count": len(rows),
            "dimension": len(rows[0]["embedding"]) if rows[0].get("embedding") else 0,
            "data": [
                {
                    "prompt_id": r["prompt_id"],
                    "participant_id": r["participant_id"],
                    "turn": r["turn"],
                    "embedding": r["embedding"],
                }
                for r in rows
            ],
        }
        emb_path = output_path / "embeddings.json"
        with open(emb_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, ensure_ascii=False)
        print(f"✅ 임베딩 벡터 저장: {emb_path}")

        # 3. 요약 통계
        from collections import Counter
        stats = {
            "total_rows": len(rows),
            "participants": len(set(r["participant_id"] for r in rows)),
            "prompt_type_dist": dict(Counter(r["prompt_type"] for r in rows)),
            "group_dist": dict(Counter(r["participant_group"] for r in rows)),
            "avg_char_count": round(sum(int(r["char_count"]) for r in rows) / len(rows), 1),
            "embedding_dim": len(rows[0]["embedding"]) if rows[0].get("embedding") else 0,
        }
        stats_path = output_path / "summary_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 요약 통계 저장: {stats_path}")

    print("\n🎉 파이프라인 완료!")


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="프롬프트 텍스트 임베딩 파이프라인")
    parser.add_argument("--input", default="data/prompt_250318.csv", help="입력 CSV 경로")
    parser.add_argument("--output-dir", default="output/", help="출력 디렉토리")
    parser.add_argument("--api-key", default=None, help="OpenAI API 키")
    parser.add_argument("--model", default="text-embedding-3-large", help="임베딩 모델")
    parser.add_argument("--skip-embedding", action="store_true", help="전처리만 실행")
    args = parser.parse_args()

    # API 키 확인
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.skip_embedding:
        print("⚠️  OpenAI API 키가 필요합니다.")
        print("   --api-key YOUR_KEY 또는 OPENAI_API_KEY 환경변수를 설정하세요.")
        print("   전처리만 실행하려면 --skip-embedding 옵션을 사용하세요.")
        return

    # 실행
    rows = load_csv(args.input)
    rows = preprocess(rows)

    if not args.skip_embedding:
        rows = run_embedding(rows, api_key, args.model)

    save_results(rows, args.output_dir)


if __name__ == "__main__":
    main()
