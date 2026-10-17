# =========================================================
# 파일 경로 설정
# =========================================================
# 한국도로교통공단 결빙 교통사고 합성데이터 20250923
DATA_FILE_PATH = "traffic_20250924_utf8.csv"

# =========================================================
# 컬럼명 상수 정의
# =========================================================
COL_MONTH       = "발생월"
COL_REGION      = "발생지역"
COL_GENDER      = "가해자성별"
COL_AGE_GROUP   = "가해자연령대"
COL_VEHICLE     = "가해자차종"
COL_TIME_SLOT   = "시간대"
COL_ACCIDENT    = "사고유형"

COL_DEATHS      = "사망자수"
COL_SERIOUS     = "중상자수"
COL_MINOR       = "경상자수"
COL_REPORTED    = "부상신고자수"


# =========================================================
# 파일 로딩
# =========================================================
import pandas as pd
import os as os

def load_raw_data(file_path: str = DATA_FILE_PATH) -> pd.DataFrame:
    """지정한 경로의 데이터를 인코딩 예외를 처리하여 로드한다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없다: {file_path}")

    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="cp949")