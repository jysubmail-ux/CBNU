import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


file_path = 'dataset/Salary_Data.csv'
dataset = pd.read_csv(file_path)

print("[데이터 정보]")
print(dataset.info())
print("\n- 데이터 상위 5행 -")
print(dataset.head())


X = dataset.iloc[:, :-1].values # 모든행 + 마지막열 제외값 : 독립변수
y = dataset.iloc[:, -1].values # 모든 행 , 마지막열 : 종속변수


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

#모델 생성 및 학습 , 내부적으로 최소자승법 시행
model = LinearRegression()  # OLS 방식 모델 정의
model.fit(X_train, y_train) # OLS 계산 실행해서 최적값 찾기

#결과 예측.
y_pred = model.predict(X_test)

#평가 함수
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- [실습 결과] ---")
print(f"기울기(Weight, w): {model.coef_[0]:.2f}")
print(f"절편(Bias, b): {model.intercept_:.2f}")
print(f"RMSE: {rmse:.2f}") # 예측이 실제값에서 평균적으로 얼마나 틀렸는지 (Root Mean Square Error)
print(f"R-squared: {r2:.4f}") # 모델이 데이터를 얼마나 잘 설명하는지 > r2 score


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='darkorange', label='Actual Data')  # 실제 데이터
plt.plot(X, model.predict(X), color='royalblue', linewidth=2, label='OLS Line')  # 회귀선
plt.title('Salary vs Experience (OLS Regression)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()