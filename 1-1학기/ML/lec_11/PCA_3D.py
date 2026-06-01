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
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
pca_df['target'] = y.values

pc1_ratio, pc2_ratio, pc3_ratio , pc4_ratio = pca.explained_variance_ratio_

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

df_0 = pca_df[pca_df['target'] == 0]
df_1 = pca_df[pca_df['target'] == 1]

ax.scatter(df_0['PC1'], df_0['PC2'], df_0['PC3'], color='steelblue', label='Benign (0)', alpha = 0.7)
ax.scatter(df_1['PC1'], df_1['PC2'], df_1['PC3'], color='indianred', label='Malignant (1)', alpha = 0.7)

ax.set_title("PCA Projection:30D to 4D")
ax.set_xlabel(f"PC1 ({pc1_ratio:.1%}")
ax.set_ylabel(f"PC2 ({pc2_ratio:.1%}")
ax.set_zlabel(f"PC3 ({pc3_ratio:.1%}")
ax.legend()

plt.show()

print(f"PC1:원본 정보의 {pc1_ratio:.1%} 보존")
print(f"PC2:원본 정보의 {pc2_ratio:.1%} 보존")
print(f"PC3:원본 정보의 {pc3_ratio:.1%} 보존")
print(f"PC4:원본 정보의 {pc4_ratio:.1%} 보존")

sum_ratio = pc1_ratio + pc2_ratio + pc3_ratio + pc4_ratio
print(f"결론 : 4개의 차원으로 원본 30차원 정보의 총 {sum_ratio:.1%} 보존 완료")