import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("11_data.csv")
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
df['target'] = df['diagnosis'].map({'B':0, 'M':1})
X = df.drop(['diagnosis', 'target'], axis=1, errors='ignore')
y = df['target']

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['target'] = y.values

pc1_ratio, pc2_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(8,5))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='target', palette='Set1', alpha=0.7)
plt.title("PCA Projection: 30D to 2D")
plt.xlabel(f"PC1 (Variance: {pc1_ratio:.1%})")
plt.xlabel(f"PC2 (Variance: {pc2_ratio:.1%})")
plt.show()

print(f"PC1:원본 정보의 {pc1_ratio:.1%} 보존")
print(f"PC2:원본 정보의 {pc2_ratio:.1%} 보존")
print(f"결론 : 2개의 차원으로 원본 30차원 정보의 총 {pc1_ratio + pc2_ratio:.1%} 보존 완료")