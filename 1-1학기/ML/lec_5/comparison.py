import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataset/Salary_Data.csv')
X,y = df[['YearsExperience']], df['Salary']
scalar = StandardScaler()
X_s = scalar.fit_transform(X)

sgd_stochastic = SGDRegressor(max_iter=1, learning_rate='constant', eta0=0.01, random_state=42)

sgd_minibatch = SGDRegressor(learning_rate='constant', eta0=0.01, random_state=42)

sgd_batch = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.01, random_state=42)

plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
sgd_stochastic.partial_fit(X_s[:1], y[:1].values.ravel())
plt.scatter(X_s,y,color='orange', alpha=0.3)
plt.plot(X_s, sgd_stochastic.predict(X_s), color='red', label='Stochastic (1 sample)')
plt.title("1. sgd_stochastic GD", fontsize=13, color='red')
plt.legend()

plt.subplot(1,3,2)
sgd_minibatch.partial_fit(X_s[:5], y[:5].values.ravel())
plt.scatter(X_s,y,color='orange', alpha=0.3)
plt.plot(X_s, sgd_minibatch.predict(X_s), color='blue', label='Mini-batch (5 sample)')
plt.title("1. Mini-batch GD", fontsize=13, color='blue')
plt.legend()

plt.subplot(1,3,3)
sgd_batch.fit(X_s, y)
plt.scatter(X_s,y,color='orange', alpha=0.3)
plt.plot(X_s, sgd_batch.predict(X_s), color='green', label='batch (ALL sample)')
plt.title("1. Full-batch GD", fontsize=13, color='green')
plt.legend()

plt.tight_layout()
plt.show()