import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



df = pd.read_csv("11_data.csv")
df = df.drop(labels=['id', 'Unnamed: 32'], axis=1, errors='ignore')
df['target'] = df['diagnosis'].map({'B': 0, 'M': 1})
X = df.drop(labels=['diagnosis', 'target'], axis=1, errors='ignore')



X_scaled = StandardScaler().fit_transform(X)
pca = PCA()
pca.fit(X_scaled)



variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)



plt.figure(figsize=(10, 6))
x_components = range(1, len(variance_ratio) + 1)



plt.bar(x_components, variance_ratio, alpha=0.5, color='steelblue', label='Individual Variance')

plt.plot(
    x_components,
    cumulative_variance,
    marker='o',
    color='indianred',
    linewidth=2,
    label='Cumulative Variance'
)

plt.title("Scree Plot: Find the 'Elbow' Point!")
plt.xlabel("Number of Principal Components (PC)")
plt.ylabel("Explained Variance Ratio (0.0 to 1.0)")

plt.xticks(range(1, 32, 2))
plt.legend(loc='center right')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



print("\n [주요 PC 개수를 느낀 정보 보존량]")
print("-" * 50)

print(f"✔ PC 2개 사용 시: {cumulative_variance[1]:.1%} 보존")
print(f"✔ PC 4개 사용 시: {cumulative_variance[3]:.1%} 보존, 변화량 : {cumulative_variance[3] - cumulative_variance[2]:.1%}")
print(f"✔ PC 5개 사용 시: {cumulative_variance[4]:.1%} 보존, 변화량 : {cumulative_variance[4] - cumulative_variance[3]:.1%}")
print(f"✔ PC 6개 사용 시: {cumulative_variance[5]:.1%} 보존, 변화량 : {cumulative_variance[5] - cumulative_variance[4]:.1%}")
print(f"✔ PC 7개 사용 시: {cumulative_variance[6]:.1%} 보존, 변화량 : {cumulative_variance[6] - cumulative_variance[5]:.1%}")
print(f"✔ PC 10개 사용 시: {cumulative_variance[9]:.1%} 보존, 변화량 : {cumulative_variance[9] - cumulative_variance[8]:.1%}")

print("-" * 50)

print("\n [PC 개수별 분류 Accuracy]")
print("-" * 50)

pc_list = [2, 4, 5, 6, 7, 10]

accuracy_results = []

for n_pc in pc_list:

    # PCA 차원축소
    pca_model = PCA(n_components=n_pc)
    X_pca = pca_model.fit_transform(X_scaled)

    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        df['target'],
        test_size=0.2,
        random_state=42,
        stratify=df['target']
    )

    # 분류 모델
    model = LogisticRegression(
        max_iter=5000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # Accuracy 계산
    acc = accuracy_score(
        y_test,
        y_pred
    )

    accuracy_results.append(acc)

    print(
        f" PC {n_pc}개 "
        f"(정보보존량 {cumulative_variance[n_pc-1]:.1%}) "
        f"→ Accuracy : {acc:.2%}"
    )

    print(classification_report(
        y_test,
        y_pred
    ))

print("-" * 50)

