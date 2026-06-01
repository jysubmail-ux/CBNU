import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt



train_df = pd.read_csv("datasets/Train.csv")
test_df = pd.read_csv("datasets/test.csv")



for df in [train_df, test_df]:
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)



for df in [train_df, test_df]:
    df["temperature"] = df["temperature"] - 273.15



train_df["is_holiday"] = train_df["is_holiday"].fillna("None")
test_df["is_holiday"] = test_df["is_holiday"].fillna("None")



feature_cols = [
    "is_holiday",
    "temperature",
    "rain_p_h",
    "wind_speed",
    "clouds_all",
    "hour",
    "hour_sin",
    "hour_cos",
    "is_weekend"
]

categorical_features = ["is_holiday"]
numeric_features = [col for col in feature_cols if col not in categorical_features]

X = train_df[feature_cols]
y = train_df["traffic_volume"]


X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

reg_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=7))
])

reg_pipe.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_pipe.predict(X_val_reg)

rmse = np.sqrt(mean_squared_error(y_val_reg, y_pred_reg))
r2 = r2_score(y_val_reg, y_pred_reg)



q1 = y.quantile(0.33)
q2 = y.quantile(0.66)
margin = (q2 - q1) * 0.1

def make_class(x):
    if x <= q1 - margin:
        return 0
    elif x <= q2 - margin:
        return 1
    elif x >= q2 + margin:
        return 2
    else:
        return np.nan

y_cls = y.apply(make_class)

valid_idx = y_cls.notna()

X_clean = X[valid_idx]
y_clean = y_cls[valid_idx]

X_train, X_val, y_train_cls, y_val_cls = train_test_split(
    X_clean, y_clean, test_size=0.2, shuffle=False
)



logistic_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

logistic_pipe.fit(X_train, y_train_cls)

y_pred_log = logistic_pipe.predict(X_val)

proba_log = logistic_pipe.predict_proba(X_val)
confidence_log = np.max(proba_log, axis=1)
mask_log = confidence_log > 0.7

y_val_log_filtered = y_val_cls[mask_log]
y_pred_log_filtered = y_pred_log[mask_log]

acc_log_filtered = accuracy_score(y_val_log_filtered, y_pred_log_filtered)



knn_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsClassifier(n_neighbors=7, weights="distance"))
])

knn_pipe.fit(X_train, y_train_cls)

y_pred_knn = knn_pipe.predict(X_val)

proba_knn = knn_pipe.predict_proba(X_val)
confidence_knn = np.max(proba_knn, axis=1)
mask_knn = confidence_knn > 0.7

y_val_knn_filtered = y_val_cls[mask_knn]
y_pred_knn_filtered = y_pred_knn[mask_knn]

acc_knn_filtered = accuracy_score(y_val_knn_filtered, y_pred_knn_filtered)



print("\n===== 최종 모델 성능 =====")

print("\n[회귀]")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.4f}")

print("\n[분류 - confidence filtering 적용]")
print(f"Logistic Accuracy: {acc_log_filtered:.4f}")
print(f"KNN Accuracy     : {acc_knn_filtered:.4f}")



labels = ["Low", "Medium", "High"]

plt.figure(figsize=(12,5))

# Logistic
plt.subplot(1,2,1)
cm_log = confusion_matrix(y_val_log_filtered, y_pred_log_filtered)
sns.heatmap(cm_log, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels)
plt.title("Logistic (Filtered)")

# KNN
plt.subplot(1,2,2)
cm_knn = confusion_matrix(y_val_knn_filtered, y_pred_knn_filtered)
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels)
plt.title("KNN (Filtered)")

plt.tight_layout()
plt.show()



model = logistic_pipe.named_steps["model"]
pre = logistic_pipe.named_steps["preprocessor"]

feature_names = pre.get_feature_names_out()
coef = model.coef_

coef_df = pd.DataFrame(coef.T, index=feature_names)
coef_df["importance"] = coef_df.abs().mean(axis=1)

top = coef_df.sort_values(by="importance", ascending=False).head(10)

plt.figure(figsize=(8,6))
plt.barh(top.index, top["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.show()


df_plot = pd.DataFrame({
    "hour": X_val_reg["hour"],
    "actual": y_val_reg.values,
    "pred": y_pred_reg
})


df_group = df_plot.groupby("hour").mean()

plt.figure(figsize=(10,6))

plt.plot(df_group.index, df_group["actual"], label="Actual")
plt.plot(df_group.index, df_group["pred"], linestyle="--", label="Predicted")

plt.title("Traffic Volume by Hour (Average)")
plt.xlabel("Hour (0~23)")
plt.ylabel("Traffic Volume")

plt.xticks(range(0,24))
plt.legend()
plt.grid()
plt.show()
