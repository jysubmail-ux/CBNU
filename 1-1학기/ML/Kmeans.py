import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


# =========================================================
# 데이터 불러오기
# =========================================================

file_path = '11_data.csv'

df = pd.read_csv(file_path)


# =========================================================
# 숫자형 데이터만 선택
# =========================================================

X = df.select_dtypes(include=[np.number])


# id 컬럼 제거
if 'id' in X.columns:
    X = X.drop(columns=['id'])


# 전체 NaN 컬럼 제거
X = X.dropna(axis=1, how='all')


# NaN 평균값 대체
X = X.fillna(X.mean())


# =========================================================
# 데이터 정규화
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================================================
# 결과 저장용 리스트
# =========================================================

k_list = []

silhouette_list = []

sse_list = []

kmeans_list = []


# =========================================================
# K-Means 반복 수행
# =========================================================

print("\nK-Means 결과")
print("=" * 50)

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)


    # ---------------------------------------------
    # 실루엣 점수
    # ---------------------------------------------

    sil_score = silhouette_score(
        X_scaled,
        labels
    )


    # ---------------------------------------------
    # SSE
    # ---------------------------------------------

    sse = kmeans.inertia_


    # 결과 저장
    k_list.append(k)

    silhouette_list.append(sil_score)

    sse_list.append(sse)

    kmeans_list.append(labels)


    # 결과 출력
    print("-" * 50)

    print(f"K = {k}")

    print(f"실루엣 점수 : {sil_score:.4f}")

    print(f"SSE         : {sse:.4f}")


# =========================================================
# 결과 표 출력
# =========================================================

result_df = pd.DataFrame({

    'K': k_list,

    'Silhouette Score': silhouette_list,

    'SSE': sse_list

})

print("\n")
print("=" * 50)
print("K-Means 결과 요약")
print("=" * 50)

print(result_df)


# =========================================================
# 최적 K 선택
# 실루엣 기준
# =========================================================

best_index = np.argmax(silhouette_list)

best_k = k_list[best_index]

best_score = silhouette_list[best_index]

best_labels = kmeans_list[best_index]


print("\n")
print("=" * 50)
print("최적 K 결과")
print("=" * 50)

print(f"최적 K 값 : {best_k}")

print(f"최대 실루엣 점수 : {best_score:.4f}")


# =========================================================
# PCA 차원 축소
# =========================================================

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)


# =========================================================
# K별 PCA 시각화
# =========================================================

for idx, k in enumerate(k_list):

    plt.figure(figsize=(7, 5))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kmeans_list[idx],
        cmap='viridis',
        s=30,
        edgecolor='k'
    )

    plt.title(
        f'K-Means Clustering Result (K={k})',
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel('Principal Component 1 (PC1)')

    plt.ylabel('Principal Component 2 (PC2)')

    plt.grid(
        linestyle='--',
        alpha=0.5
    )

    plt.tight_layout()

    plt.show()


# =========================================================
# SSE 그래프
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_list,
    sse_list,
    marker='o',
    linewidth=2
)

plt.title(
    'K-SSE Result',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('K Value')

plt.ylabel('SSE')

plt.xticks(k_list)

plt.grid(
    linestyle='--',
    alpha=0.5
)

plt.show()


# =========================================================
# 실루엣 점수 그래프
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_list,
    silhouette_list,
    marker='o',
    linewidth=2
)

plt.title(
    'K-Silhouette Score Result',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('K Value')

plt.ylabel('Silhouette Score')

plt.xticks(k_list)

plt.grid(
    linestyle='--',
    alpha=0.5
)

plt.show()


# =========================================================
# 최적 K 군집 시각화
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=best_labels,
    cmap='viridis',
    s=35,
    edgecolor='k'
)

plt.title(
    f'K-Means Clustering Result (K={best_k})',
    fontsize=14,
    fontweight='bold'
)

plt.xlabel('Principal Component 1 (PC1)')

plt.ylabel('Principal Component 2 (PC2)')

plt.grid(
    linestyle='--',
    alpha=0.5
)

plt.tight_layout()

plt.show()


# =========================================================
# 군집별 데이터 개수
# =========================================================

print("\n")
print("=" * 50)
print(f"K={best_k} 군집별 데이터 개수")
print("=" * 50)

unique, counts = np.unique(
    best_labels,
    return_counts=True
)

for u, c in zip(unique, counts):

    print(f"Cluster {u} : {c}개")