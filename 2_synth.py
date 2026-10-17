from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import config


def generate_synth_data() -> str:
    """설계된 순서, 방법, 제약조건(minbucket=5)을 반영하여 합성데이터를 순차 생성하고 파일명을 반환한다."""
    # 1. 원본 데이터 로드
    orig_data = config.load_raw_data(config.DATA_FILE_PATH)

    # 2. 순서 정의 (환경 -> 주체 -> 사건 -> 결과) 총 10개 컬럼
    col_order = [
        config.COL_TIME_SLOT,  # 첫 변수: sample 방식 (시간대)
        config.COL_REGION,     # 발생지역
        config.COL_GENDER,     # 가해자성별
        config.COL_AGE_GROUP,  # 가해자연령대
        config.COL_VEHICLE,    # 가해자차종
        config.COL_ACCIDENT,   # 사고유형
        config.COL_DEATHS,     # 사망자수
        config.COL_SERIOUS,    # 중상자수
        config.COL_MINOR,      # 경상자수
        config.COL_REPORTED,   # 부상신고자수
    ]

    MIN_BUCKETS = 5
    RANDOM_SEED = None

    np.random.seed(None)
    n_rows = len(orig_data)
    synth_dict = {}

    print("=== 순환 CART 합성 데이터 생성 시작 ===")

    # 3. 순차 생성 (첫 컬럼은 sample, 이후는 CART minbucket=5)
    for i, col in enumerate(col_order):
        if i == 0:
            # 1단계: 시간대 (sample 무작위 복원추출)
            synth_dict[col] = orig_data[col].sample(n=n_rows, replace=True, random_state=RANDOM_SEED).values
            print(f"[{i+1}/{len(col_order)}] {col} 생성 완료 (sample)")
            continue

        # 이전까지 생성된 컬럼들을 조건(X)으로 설정
        prev_cols = col_order[:i]
        X_train = pd.get_dummies(orig_data[prev_cols], drop_first=False)
        y_train = orig_data[col]

        # 합성 데이터용 조건(X_synth) 정렬
        df_prev_synth = pd.DataFrame({c: synth_dict[c] for c in prev_cols})
        X_synth = pd.get_dummies(df_prev_synth, drop_first=False)
        X_synth = X_synth.reindex(columns=X_train.columns, fill_value=0)

        # 2단계 이후: CART 모델링 (minbucket = 5 반영)
        # 수치형 여부를 기준으로 명확히 분기 (문자열 타입 변화 대응)
        if not pd.api.types.is_numeric_dtype(y_train):
            # 범주형 컬럼: 분류 나무 (DecisionTreeClassifier)
            clf = DecisionTreeClassifier(min_samples_leaf=MIN_BUCKETS, random_state=RANDOM_SEED)
            clf.fit(X_train, y_train.astype(str))

            # 조건부 확률 분포 기반 무작위 추출
            probs = clf.predict_proba(X_synth)
            classes = clf.classes_
            synth_values = [
                np.random.choice(classes, p=p) if p.sum() > 0 else np.random.choice(classes)
                for p in probs
            ]
            synth_dict[col] = np.array(synth_values)
        else:
            # 수치형 컬럼: 회귀 나무 (DecisionTreeRegressor)
            reg = DecisionTreeRegressor(min_samples_leaf=MIN_BUCKETS, random_state=RANDOM_SEED)
            reg.fit(X_train, y_train)
            preds = reg.predict(X_synth)
            # 정수 변환 및 음수 방지 (사후 제약 규칙)
            synth_dict[col] = np.clip(np.round(preds), 0, None).astype(int)

        print(f"[{i+1}/{len(col_order)}] {col} 생성 완료 (cart, minbucket={MIN_BUCKETS})")

    # 4. 데이터프레임 변환 및 파일 저장
    df_synthetic = pd.DataFrame(synth_dict)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_path = f"synth_traffic_{timestamp}.csv"
    df_synthetic.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n생성 완료: {output_path} ({len(df_synthetic)}건 저장)")
    return output_path


if __name__ == "__main__":
    saved_file = generate_synth_data()