import glob
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, ks_2samp
from scipy.stats import chi2_contingency, f_oneway

import config


def get_latest_synth_file() -> str:
    """디렉터리 내 가장 최근 생성된 합성 데이터 파일명을 탐색한다."""
    synth_files = glob.glob("synth_traffic_*.csv")
    if not synth_files:
        raise FileNotFoundError(
            "생성된 합성 데이터(synth_traffic_*.csv)를 찾을 수 없다."
        )
    return max(synth_files, key=os.path.getctime)


def load_datasets(synth_path: str = None):
    """config.load_raw_data를 활용하여 원본 및 검증 대상 합성 데이터를 로드한다."""
    # 1. 원본 데이터 로드
    df_orig = config.load_raw_data()

    # 2. 합성 데이터 경로 결정 및 로드
    if synth_path is None:
        synth_path = get_latest_synth_file()

    print(f"▶ 원본 데이터 파일: {config.DATA_FILE_PATH}")
    print(f"▶ 검증 대상 합성 파일: {synth_path}")

    df_synth = config.load_raw_data(synth_path)
    return df_orig, df_synth


# ==========================================
# 1. 유용성(Utility) 검증
# ==========================================

def evaluate_1d_distribution(df_orig, df_synth, cat_cols, num_cols):
    """[가이드라인 기준] 일차원 분포 유사성 검증

    판정 기준:
    - JSD: 0.05 이하 (통과)
    - 카이제곱 통계량: 범주 수(k) 이하 (통과)
    - KS 통계량(D): 0.10(10%) 미만 (통과)
    """
    print("\n" + "=" * 50)
    print(" [유용성 1] 일차원 분포 유사성")
    print("=" * 50)

    # 1. 범주형 컬럼 분포 유사도
    print("1) 범주형 컬럼 분포 유사도:")
    for col in cat_cols:
        all_cats = sorted(
            list(set(df_orig[col].unique()) | set(df_synth[col].unique()))
        )
        k = len(all_cats)  # 범주 수

        # JSD 계산
        p = (
            df_orig[col]
            .value_counts(normalize=True)
            .reindex(all_cats, fill_value=0)
            .values
        )
        q = (
            df_synth[col]
            .value_counts(normalize=True)
            .reindex(all_cats, fill_value=0)
            .values
        )
        jsd = jensenshannon(p, q, base=2) ** 2
        jsd_status = "통과" if jsd <= 0.05 else "미달"

        # 카이제곱 통계량 계산
        f_obs = df_synth[col].value_counts().reindex(all_cats, fill_value=0)
        f_exp = (
            df_orig[col]
            .value_counts(normalize=True)
            .reindex(all_cats, fill_value=0)
            * len(df_synth)
        )
        valid = f_exp > 0
        chi2_stat, _ = chisquare(f_obs[valid], f_exp[valid])
        chi2_status = "통과" if chi2_stat <= k else "미달"

        print(
            f"  - {col:<12}: JSD = {jsd:.4f}({jsd_status}) | 카이제곱 = {chi2_stat:6.2f}({chi2_status}, 기준: ≤{k})"
        )

    # 2. 수치형 컬럼 분포 유사도
    print("\n2) 수치형 컬럼 분포 유사도:")
    for col in num_cols:
        # KS 통계량(D) 계산
        ks_stat, _ = ks_2samp(df_orig[col], df_synth[col])
        ks_status = "통과" if ks_stat < 0.10 else "미달"

        # 수치형 JSD 계산
        min_v = min(df_orig[col].min(), df_synth[col].min())
        max_v = max(df_orig[col].max(), df_synth[col].max())
        bins = max(5, int(max_v - min_v + 1))

        p_hist, bin_edges = np.histogram(
            df_orig[col], bins=bins, range=(min_v, max_v + 1), density=True
        )
        q_hist, _ = np.histogram(df_synth[col], bins=bin_edges, density=True)

        p = p_hist / np.sum(p_hist) if np.sum(p_hist) > 0 else p_hist
        q = q_hist / np.sum(q_hist) if np.sum(q_hist) > 0 else q_hist
        jsd = jensenshannon(p, q, base=2) ** 2
        jsd_status = "통과" if jsd <= 0.05 else "미달"

        print(
            f"  - {col:<12}: JSD = {jsd:.4f}({jsd_status}) | KS 통계량(D) = {ks_stat:.4f}({ks_status}, 기준: <0.10)"
        )


# ========================================== 여기까지 1차원 분포 유사성


def cramers_v(x, y):
    """범주형-범주형: 크래머 V 상관계수 (0 ~ 1)"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    phi2 = chi2 / n
    if min(r, k) <= 1:
        return 0.0
    return np.sqrt(phi2 / (min(r, k) - 1))


def eta_squared(cat_var, num_var):
    """범주형-수치형: ANOVA 결정계수 (SSB/SST, 0 ~ 1)"""
    groups = [
        group.dropna().values for _, group in num_var.groupby(cat_var)
    ]
    if len(groups) <= 1:
        return 0.0
    # 전체 제곱합(SST) 및 그룹 간 제곱합(SSB) 계산
    all_vals = num_var.dropna().values
    grand_mean = np.mean(all_vals)
    sst = np.sum((all_vals - grand_mean) ** 2)
    if sst == 0:
        return 0.0
    ssb = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
    return np.sqrt(ssb / sst)  # 비교를 위해 상관비(sqrt(eta^2)) 형태로 변환


def compute_mixed_correlation_matrix(df, cat_cols, num_cols):
    """모든 변수 쌍(수치-수치, 범주-범주, 범주-수치)의 통합 상관 행렬 생성"""
    all_cols = cat_cols + num_cols
    corr_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)

    for i, col1 in enumerate(all_cols):
        for j, col2 in enumerate(all_cols):
            if i == j:
                corr_matrix.loc[col1, col2] = 1.0
            elif i > j:
                corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1]
            else:
                # 1. 수치형 - 수치형 (피어슨)
                if col1 in num_cols and col2 in num_cols:
                    val = df[col1].corr(df[col2])
                # 2. 범주형 - 범주형 (크래머 V)
                elif col1 in cat_cols and col2 in cat_cols:
                    val = cramers_v(df[col1], df[col2])
                # 3. 범주형 - 수치형 (상관비)
                elif col1 in cat_cols and col2 in num_cols:
                    val = eta_squared(df[col1], df[col2])
                else:
                    val = eta_squared(df[col2], df[col1])

                corr_matrix.loc[col1, col2] = (
                    0.0 if np.isnan(val) else abs(val)
                )

    return corr_matrix

def evaluate_2d_correlation(df_orig, df_synth, cat_cols, num_cols):
    """[가이드라인 기준] 2차원 관계 유사성 종합 검증

    판정 기준:
    - 상관계수 오차 범위(수치, 범주, 복합): 최대 오차 <= 0.10 (통과)
    - 종합 지표 sigma_D: <= 0.15 (통과)
    """
    print("\n" + "=" * 50)
    print(" [유용성 2] 2차원 관계 유사성 (전체 속성 결합)")
    print("=" * 50)

    # 1. 통합 상관계수 행렬 산출
    corr_orig = compute_mixed_correlation_matrix(df_orig, cat_cols, num_cols)
    corr_synth = compute_mixed_correlation_matrix(
        df_synth, cat_cols, num_cols
    )

    # 2. 미달 항목 수집용 리스트
    failed_num = []
    failed_cat = []
    failed_mixed = []

    # 1) 수치형 - 수치형 (피어슨)
    diffs_num = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            diff = abs(corr_orig.loc[c1, c2] - corr_synth.loc[c1, c2])
            diffs_num.append(diff)
            if diff > 0.10:
                failed_num.append(
                    (
                        f"{c1} - {c2}",
                        diff,
                        corr_orig.loc[c1, c2],
                        corr_synth.loc[c1, c2],
                    )
                )

    # 2) 범주형 - 범주형 (크래머 V)
    diffs_cat = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            c1, c2 = cat_cols[i], cat_cols[j]
            diff = abs(corr_orig.loc[c1, c2] - corr_synth.loc[c1, c2])
            diffs_cat.append(diff)
            if diff > 0.10:
                failed_cat.append(
                    (
                        f"{c1} - {c2}",
                        diff,
                        corr_orig.loc[c1, c2],
                        corr_synth.loc[c1, c2],
                    )
                )

    # 3) 범주형 - 수치형 (ANOVA 상관비)
    diffs_mixed = []
    for c1 in cat_cols:
        for c2 in num_cols:
            diff = abs(corr_orig.loc[c1, c2] - corr_synth.loc[c1, c2])
            diffs_mixed.append(diff)
            if diff > 0.10:
                failed_mixed.append(
                    (
                        f"{c1} - {c2}",
                        diff,
                        corr_orig.loc[c1, c2],
                        corr_synth.loc[c1, c2],
                    )
                )

    # 3. 요약 통계량 계산 헬퍼
    def get_stats(diffs):
        min_v = np.min(diffs)
        med_v = np.median(diffs)
        mean_v = np.mean(diffs)
        max_v = np.max(diffs)
        status = "통과" if max_v <= 0.10 else "미달"
        return min_v, med_v, mean_v, max_v, status

    min_n, med_n, mean_n, max_n, stat_n = get_stats(diffs_num)
    min_c, med_c, mean_c, max_c, stat_c = get_stats(diffs_cat)
    min_m, med_m, mean_m, max_m, stat_m = get_stats(diffs_mixed)

    # 4. 전체 차이 행렬의 표준편차 (sigma_D)
    diff_matrix = (corr_orig - corr_synth).values.astype(float)
    sigma_d = np.std(diff_matrix)
    stat_sigma = "통과" if sigma_d <= 0.15 else "미달"

    print("1) 변수 유형별 상관관계 절대오차 범위 [최소 / 중앙값 / 평균 / 최대]:")
    print(
        f"  - 수치형 변수 간 (피어슨)     : [{min_n:.4f} / {med_n:.4f} / {mean_n:.4f} / {max_n:.4f}] ({stat_n}, 기준: 최대 ≤0.10)"
    )
    print(
        f"  - 범주형 변수 간 (크래머 V)   : [{min_c:.4f} / {med_c:.4f} / {mean_c:.4f} / {max_c:.4f}] ({stat_c}, 기준: 최대 ≤0.10)"
    )
    print(
        f"  - 범주형-수치형 간 (ANOVA 상관비): [{min_m:.4f} / {med_m:.4f} / {mean_m:.4f} / {max_m:.4f}] ({stat_m}, 기준: 최대 ≤0.10)"
    )

    # 5. 조합별 통과/미달 건수 요약 및 세부 내역 출력
    total_n, fail_n = len(diffs_num), len(failed_num)
    total_c, fail_c = len(diffs_cat), len(failed_cat)
    total_m, fail_m = len(diffs_mixed), len(failed_mixed)

    print("\n   [변수 조합별 통과 현황 (기준: 오차 ≤ 0.10)]")
    print(
        f"   • 수치형 변수 간     : 총 {total_n}개 조합 중 통과 {total_n - fail_n}개 / 미달 {fail_n}개"
    )
    if failed_num:
        for pair, diff, orig, synth in failed_num:
            print(
                f"     - {pair:<20}: 오차 {diff:.4f} (원본 {orig:+.4f} vs 합성 {synth:+.4f})"
            )

    print(
        f"   • 범주형 변수 간     : 총 {total_c}개 조합 중 통과 {total_c - fail_c}개 / 미달 {fail_c}개"
    )
    if failed_cat:
        for pair, diff, orig, synth in failed_cat:
            print(
                f"     - {pair:<20}: 오차 {diff:.4f} (원본 {orig:.4f} vs 합성 {synth:.4f})"
            )

    print(
        f"   • 범주형 - 수치형 간 : 총 {total_m}개 조합 중 통과 {total_m - fail_m}개 / 미달 {fail_m}개"
    )
    if failed_mixed:
        for pair, diff, orig, synth in failed_mixed:
            print(
                f"     - {pair:<20}: 오차 {diff:.4f} (원본 {orig:.4f} vs 합성 {synth:.4f})"
            )

    print("\n2) 전체 2차원 관계 종합 지표:")
    print(
        f"  - 상관계수 차이 표준편차 (sigma_D): {sigma_d:.4f} ({stat_sigma}, 기준: ≤0.15)"
    )

def evaluate_indistinguishability(df_orig, df_synth):
    """[가이드라인 기준] 구별 불가능성 3대 지표 (pMSE, SPECKS, PO50) 통합 검증"""
    print("\n" + "=" * 50)
    print(" [유용성 3] 구별 불가능성 (Indistinguishability)")
    print("=" * 50)

    # 1. 데이터 결합 및 레이블링 (원본: 0, 합성: 1)
    df_o = df_orig.copy()
    df_s = df_synth.copy()
    df_o["is_synth"] = 0
    df_s["is_synth"] = 1

    combined = pd.concat([df_o, df_s], ignore_index=True)
    X = pd.get_dummies(combined.drop(columns=["is_synth"]), drop_first=True)
    y = combined["is_synth"].values

    # 2. 구별 판별 모형 학습 (로지스틱 회귀 기준)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    # 각 레코드가 '합성데이터'일 확률 예측 (p_hat)
    probs = model.predict_proba(X)[:, 1]
    N = len(y)
    c = 0.5  # 1:1 결합 시 이론적 기댓값

    # --------------------------------------------------
    # 1) pMSE & pMSE Ratio
    # --------------------------------------------------
    pmse = np.mean((probs - c) ** 2)

    # 이론적 기댓값: (k - 1) * (1 - c)^2 / N  (k: 모델 파라미터/특성 수)
    k = X.shape[1] + 1
    expected_pmse = (k - 1) * ((1 - c) ** 2) * c / N
    pmse_ratio = pmse / expected_pmse if expected_pmse > 0 else 0.0

    stat_pmse = "통과" if pmse_ratio <= 2.0 else "미달"

    # --------------------------------------------------
    # 2) SPECKS (Kolmogorov-Smirnov distance between probs)
    # --------------------------------------------------
    probs_orig = probs[y == 0]
    probs_synth = probs[y == 1]
    specks_stat, _ = ks_2samp(probs_orig, probs_synth)
    stat_specks = "통과" if specks_stat <= 0.10 else "미달"

    # --------------------------------------------------
    # 3) PO50 (Percent Over 0.5)
    # --------------------------------------------------
    po50 = np.mean(probs > 0.5)
    stat_po50 = "통과" if 0.45 <= po50 <= 0.55 else "미달"

    # --------------------------------------------------
    # 결과 출력
    # --------------------------------------------------
    print(
        f"  - pMSE         : {pmse:.6f} (pMSE Ratio: {pmse_ratio:.2f}) ({stat_pmse}, 기준: Ratio ≤2.0)"
    )
    print(
        f"  - SPECKS       : {specks_stat:.4f} ({stat_specks}, 기준: ≤0.10)"
    )
    print(
        f"  - PO50         : {po50 * 100:.2f}% ({stat_po50}, 기준: 45%~55%)"
    )


# ==========================================
# 2. 안전성(Privacy/Risk) 검증
# ==========================================


def evaluate_single_out_risk(df_orig, df_synth):
    """구별 위험도 (Single-out Risk / 원본 동일행 비율)"""
    print("\n" + "=" * 50)
    print(" [안전성 1] 구별 위험도 (Single-out Risk)")
    print("=" * 50)

    cols = df_orig.columns.tolist()
    merged = pd.merge(df_synth, df_orig, on=cols, how="inner")
    identical_cnt = len(merged)
    risk = identical_cnt / len(df_synth)

    print(
        f"  - 원본 완전 일치 레코드 수: {identical_cnt}건 / 전체 {len(df_synth)}건"
    )
    print(f"  - 구별 위험도: {risk:.4%}")
    print("    (0%에 가까울수록 원본 복제 및 과적합 위험이 낮음)")


def evaluate_linkability_risk(
    df_orig, df_synth, quasi_identifiers, sensitive_attribute
):
    """연결 위험도 (Linkability / CAP: Correct Attribution Probability)"""
    print("\n" + "=" * 50)
    print(" [안전성 2] 연결 위험도 (CAP)")
    print("=" * 50)

    # 합성 데이터의 준식별자 결합별 민감속성 리스트 매핑
    synth_groups = (
        df_synth.groupby(quasi_identifiers)[sensitive_attribute]
        .apply(list)
        .to_dict()
    )

    cap_list = []
    for _, row in df_orig.iterrows():
        qk = tuple(row[col] for col in quasi_identifiers)
        sens_val = row[sensitive_attribute]

        if qk in synth_groups:
            matched_vals = synth_groups[qk]
            cap_i = matched_vals.count(sens_val) / len(matched_vals)
            cap_list.append(cap_i)

    avg_cap = np.mean(cap_list) if cap_list else 0.0
    print(f"  - 준식별자 결합: [{', '.join(quasi_identifiers)}]")
    print(f"  - 추론 대상 민감속성: [{sensitive_attribute}]")
    print(f"  - 평균 CAP: {avg_cap:.4f}")
    print("    (가이드라인 기준 0.5 이하 권고, 0에 가까울수록 안전)")


def evaluate_inference_risk(df_orig, df_synth, sample_size=300):
    """추론 위험도 (Inference Risk / 거리 기반 복제 검사)"""
    print("\n" + "=" * 50)
    print(" [안전성 3] 추론 위험도 (거리 기반 d_s < d_o 검사)")
    print("=" * 50)

    cols = df_orig.columns.tolist()
    combined = pd.concat([df_orig[cols], df_synth[cols]], ignore_index=True)
    combined_enc = pd.get_dummies(combined, drop_first=False).astype(float)

    n_o = len(df_orig)
    orig_enc = combined_enc.iloc[:n_o].values
    synth_enc = combined_enc.iloc[n_o:].values

    np.random.seed(42)
    s_idx = np.random.choice(
        len(synth_enc), min(sample_size, len(synth_enc)), replace=False
    )
    synth_sub = synth_enc[s_idx]

    # 1. d_s: 각 합성 레코드와 가장 가까운 원본 레코드 간 거리
    dist_s_to_o = cdist(synth_sub, orig_enc, metric="euclidean")
    min_ds = np.min(dist_s_to_o, axis=1)
    nearest_orig_idx = np.argmin(dist_s_to_o, axis=1)

    # 2. d_o: 해당 원본 레코드와 가장 가까운 다른 원본 레코드 간 거리
    count_risk = 0
    valid_count = 0
    for i, orig_target in enumerate(nearest_orig_idx):
        target_vec = orig_enc[orig_target : orig_target + 1]
        dists = cdist(target_vec, orig_enc, metric="euclidean")[0]
        dists[orig_target] = np.inf  # 자기 자신 제외
        min_do = np.min(dists)

        if min_ds[i] < min_do:
            count_risk += 1
        valid_count += 1

    inference_risk = count_risk / valid_count if valid_count > 0 else 0
    print(f"  - 추론 위험도 (d_s < d_o 비율): {inference_risk:.4f}")
    print("    (이론적 이상치 0.5 수준, 1에 가까울수록 원본 복제 과적합)")


# ==========================================
# 메인 실행 파이프라인
# ==========================================


def main():
    # 실행 시 특정 합성 파일명을 인자로 넘길 수 있음 (미지정 시 가장 최근 파일 자동 선택)
    target_synth_file = sys.argv[1] if len(sys.argv) > 1 else None
    df_orig, df_synth = load_datasets(target_synth_file)

    cat_cols = [
        config.COL_TIME_SLOT,
        config.COL_REGION,
        config.COL_GENDER,
        config.COL_AGE_GROUP,
        config.COL_VEHICLE,
        config.COL_ACCIDENT,
    ]
    num_cols = [
        config.COL_DEATHS,
        config.COL_SERIOUS,
        config.COL_MINOR,
        config.COL_REPORTED,
    ]

    # 1. 유용성 검증 수행
    evaluate_1d_distribution(df_orig, df_synth, cat_cols, num_cols)
    evaluate_2d_correlation(df_orig, df_synth, cat_cols, num_cols)
    evaluate_indistinguishability(df_orig, df_synth)

    # 2. 안전성 검증 수행
    evaluate_single_out_risk(df_orig, df_synth)

    quasi_ids = [config.COL_REGION, config.COL_GENDER, config.COL_AGE_GROUP]
    evaluate_linkability_risk(
        df_orig, df_synth, quasi_ids, sensitive_attribute=config.COL_DEATHS
    )

    evaluate_inference_risk(df_orig, df_synth, sample_size=300)
    print("\n" + "=" * 50)
    print(" 품질검증 파이프라인 검사가 완료되었습니다.")
    print("=" * 50)


if __name__ == "__main__":
    main()