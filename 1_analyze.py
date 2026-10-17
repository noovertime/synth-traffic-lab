import pandas as pd
import config

def calculate_single_qi_risk(df: pd.DataFrame, qi_cols: list[str]) -> dict:
    """단일 준식별자 조합의 결합 위험도(빈도 1 비율)를 계산한다."""
    total_records = len(df)
    group_sizes = df.groupby(qi_cols).size()
    unique_count = int((group_sizes == 1).sum())
    risk_ratio = (unique_count / total_records) * 100

    return {
        "columns": qi_cols,
        "unique_count": unique_count,
        "risk_ratio": risk_ratio,
    }


def analyze_qi_risk(df: pd.DataFrame) -> None:
    """설정된 모든 준식별자 조합에 대해 결합 위험도를 분석하고 출력한다."""
    print(f"=== 준식별자 결합 위험성 분석 : 대상 데이터 총 {len(df):,}건 ===\n")

    # =========================================================
    # 준식별자(QI) 결합 단계 정의
    # =========================================================
    QI_COMBINATIONS = {
        "2개 조합 (지역+성별)": [config.COL_REGION, config.COL_GENDER],
        "3개 조합 (지역+성별+연령대)": [config.COL_REGION, config.COL_GENDER, config.COL_AGE_GROUP],
        "4개 조합 (지역+성별+연령대+차종)": [config.COL_REGION, config.COL_GENDER, config.COL_AGE_GROUP, config.COL_VEHICLE],
        "5개 조합 (지역+성별+연령대+차종+시간대)": [config.COL_REGION, config.COL_GENDER, config.COL_AGE_GROUP, config.COL_VEHICLE, config.COL_TIME_SLOT],
    }

    for name, cols in QI_COMBINATIONS.items():
        result = calculate_single_qi_risk(df, cols)

        print(f"[{name}]")
        print(f" - 컬럼: {result['columns']}")
        print(f" - 유일 레코드(빈도 1): {result['unique_count']:,}건")
        print(f" - 결합 식별 위험률: {result['risk_ratio']:.2f}%\n")



def check_missing_values(df: pd.DataFrame) -> None:
    """컬럼별 결측치(NaN 및 공백) 현황을 점검한다."""
    print("=== 1. 결측치 점검 결과 ===")
    missing_counts = df.isnull().sum()
    total_records = len(df)

    has_missing = False
    for col, count in missing_counts.items():
        if count > 0:
            has_missing = True
            ratio = (count / total_records) * 100
            print(f" - {col}: {count:,}건 ({ratio:.2f}%) 결측")

    if not has_missing:
        print(" - 모든 컬럼에 결측치(Null/NaN) 없음 (완전 데이터)")
    print()


def check_rare_categories(df: pd.DataFrame, threshold: int = 5) -> None:
    """범주형 컬럼별로 출현 빈도가 threshold(기본 5건) 이하인 희소 범주를 점검한다."""
    print(f"=== 2. 희소 범주 점검 결과 (기준: {threshold}건 이하) ===")
    total_records = len(df)

    # 분석 대상 범주형 컬럼 정의
    CATEGORICAL_COLS = [
        config.COL_REGION,
        config.COL_GENDER,
        config.COL_AGE_GROUP,
        config.COL_VEHICLE,
        config.COL_TIME_SLOT,
        config.COL_ACCIDENT,
    ]

    for col in CATEGORICAL_COLS:
        value_counts = df[col].value_counts()
        rare_items = value_counts[value_counts <= threshold]

        if not rare_items.empty:
            print(f"[{col}] (총 범주 수: {len(value_counts)}개 / 희소 범주: {len(rare_items)}개)")
            for val, cnt in rare_items.items():
                ratio = (cnt / total_records) * 100
                print(f"   • '{val}': {cnt}건 ({ratio:.2f}%)")
        else:
            print(f"[{col}] 희소 범주 없음 (모든 범주 빈도 > {threshold}건)")
    print()


def validate_business_rules(df: pd.DataFrame) -> None:
    """교통사고 데이터의 업무 논리적 제약(Business Rules) 위반 여부를 검증하고 출력한다."""
    casualty_cols = ["사망자수", "중상자수", "경상자수", "부상신고자수"]
    motor_vehicles = ["승용차", "화물차", "승합차", "건설기계", "특수차"]

    # 1. 인명 피해 수치 중 음수(-) 존재 여부 점검
    negative_mask = (df[casualty_cols] < 0).any(axis=1)
    negative_count = int(negative_mask.sum())

    # 2. 사고 발생 건인데 총 사상자 수가 0명인 비정상 레코드 점검
    total_casualties = df[casualty_cols].sum(axis=1)
    zero_casualty_count = int((total_casualties == 0).sum())

    # 3. 면허 취득 불가 연령(10대 미만)의 자동차 운전 여부 점검
    underage_mask = (df[config.COL_VEHICLE].isin(motor_vehicles)) & (
        df[config.COL_AGE_GROUP].astype(str).str.contains("10대미만|유아|어린이", na=False)
    )
    underage_driver_count = int(underage_mask.sum())

    print("=== 논리적 제약(Business Rules) 점검 결과 ===")
    print(f"1. 인명 피해 음수(-) 레코드: {negative_count}건")
    print(f"2. 사상자 총합 0명인 레코드: {zero_casualty_count}건")
    print(f"3. 자동차(승용/화물 등) 운전 10대 미만 레코드: {underage_driver_count}건\n")



def main():
    df = config.load_raw_data(config.DATA_FILE_PATH)

    # 준식별자 결합 위험성 분석
    #analyze_qi_risk(df)

    # 결측치 분석
    #check_missing_values(df)

    # 희소범주 점검
    #check_rare_categories(df, threshold=5)

    # 논리적 제약 도출
    #validate_business_rules(df)

if __name__ == "__main__":
    main()