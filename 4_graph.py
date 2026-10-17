import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

# ==============================================================================
# 1. 원본 데이터 및 합성 데이터 로드
# ==============================================================================
# 원본 데이터 로드 (config.py 함수 사용)
df_raw = config.load_raw_data()

# 분석 대상 컬럼 설정 (config 상수 활용)
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
target_cols = cat_cols + num_cols

# 결측치 제거
df_orig = df_raw[target_cols].dropna().reset_index(drop=True)

# 생성해 둔 합성 데이터 파일 로드
synth_file_path = "synth_traffic_1017.csv"
df_synth = pd.read_csv(synth_file_path)[target_cols].dropna().reset_index(drop=True)

# ==============================================================================
# 2. 통합 인코딩 및 PCA 2차원 차원축소
# ==============================================================================
df_o = df_orig.copy()
df_s = df_synth.copy()
df_o["type"] = "Original"
df_s["type"] = "Synthetic"

combined = pd.concat([df_o, df_s], ignore_index=True)
combined_encoded = pd.get_dummies(
    combined.drop(columns=["type"]), drop_first=True
)

# 표준화 후 PCA 축소
X_scaled = StandardScaler().fit_transform(combined_encoded)
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(X_scaled)

combined["pca_x"] = pca_result[:, 0]
combined["pca_y"] = pca_result[:, 1]


loadings = pd.DataFrame(
    pca.components_.T,
    columns=["Component_1", "Component_2"],
    index=combined_encoded.columns,
)

print("=== [가로축] Component 1을 결정짓는 상위 속성 ===")
print(loadings["Component_1"].abs().sort_values(ascending=False).head(5))

print("\n=== [세로축] Component 2를 결정짓는 상위 속성 ===")
print(loadings["Component_2"].abs().sort_values(ascending=False).head(5))



# ==============================================================================
# 3. 2D 비교 산점도 시각화 및 이미지 파일 저장
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

for ax, t in zip(axes, ["Original", "Synthetic"]):
    sub = combined[combined["type"] == t]
    ax.scatter(sub["pca_x"], sub["pca_y"], alpha=0.4, s=25, c="#1f77b4")
    ax.set_title(f"{t} data", fontsize=14, fontweight="bold")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True, linestyle="--", alpha=0.5)

# 한글 폰트 깨짐 방지를 위해 영문 메인 타이틀 적용
plt.suptitle(
    "Original vs Synthetic Data (PCA 2D Projection)",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("pca_comparison.png", dpi=300)
plt.show()
print("시각화 완료: pca_comparison.png 저장됨")