import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# [Part 1 & 2: Classification Tree Comparison]
# ID3/C4.5 (Entropy) vs CART (Gini Index)


'''
iris 데이터를 기준. 꽃잎의 길이가 가장 먼저 분기
value값은 각각의 클래스 값 [setosa , versicolor, virginica]를 의미

처음 분기로 setosa 와 그외로 분기가 확실히 됨.
그 이후 분기데이터를 보면
49:1 / 45:5 이렇게 분기되어 있는걸 볼 수 있음
엔트로피의 경우 완벽하게 분기된 경우 0, 분기가 필요한 데이터의 경우 0~1의 값을 가짐.

이 값이 낮을수록 명확하게 분기가 된 것으로 파악

이후 추가 분기를 통해 구분
'''

iris = load_iris()

# Petal length, width만 사용 (시각화용)
X = iris.data[:, 2:]
y = iris.target





tree_entropy = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)
tree_entropy.fit(X, y)


tree_gini = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)
tree_gini.fit(X, y)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

# Entropy Tree
plot_tree(
    tree_entropy,
    filled=True,
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    ax=axes[0]
)
axes[0].set_title("Entropy-based Tree (ID3/C4.5 Concept)", fontsize=16)

# Gini Tree
plot_tree(
    tree_gini,
    filled=True,
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    ax=axes[1]
)
axes[1].set_title("Gini-based Tree (Standard CART)", fontsize=16)

plt.tight_layout()
plt.show()