import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

file_path = "Salary_Data.csv"
data = pd.read_csv(file_path)

X = data[['YearsExperience']]
y = data['Salary']

model = LinearRegression()
model.fit(X,y)

w1 = model.coef_[0] # 연봉 인상액
w0 = model.intercept_ # 신입 연봉


print(f"=== AI 분석 결과 ===")
print(f"공식 : Salary (w1) : {w1:.2f} * Experience + {w0:.2f}")
print(f"1) 연봉 인상액 : ${w1:.2f}")
print(f"2) 예상 초봉 : ${w0:.2f}")

y_pred = model.predict(X)
print(f"3) 정확도 (R2) : {r2_score(y, y_pred):.4f}")

X_new = pd.DataFrame([[15]], columns=['YearsExperience'])
pred_15 = model.predict(X_new)
print(f"4) 15년차 예상 연봉 : ${pred_15[0]:,.2f}")


plt.figure(figsize=(10,6))
plt.scatter(X,y, color='blue', label='Actual Data')
plt.plot(X,y_pred, color='red', linewidth=2, label='Regression Line')

plt.title("Salary vs Experience Analysis", fontsize=15)
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()