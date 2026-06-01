import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.sparsefuncs import min_max_axis


#data = pd.read_csv('datasets/housing.csv')

def load_dataset():
    baspath = Path(__file__).parent
    csv_path = baspath / "datasets" / "housing.csv"

    return pd.read_csv(csv_path)



if __name__ == "__main__":

    housing = load_dataset()
    pd.set_option('display.max_columns', None)

    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)

    print("\n Step 4: 스케이링 전 원본 데이터 출력")
    print("-" * 50)

    num_housing = housing.drop("ocean_proximity", axis=1)
    print(num_housing.describe())

    std_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()

    housing_std = std_scaler.fit_transform(num_housing)
    housing_minmax = min_max_scaler.fit_transform(num_housing)
    print("\n Step 5: 스케일링 이후 데이터 (median_income column")

    print("-" * 50)

    print(f"Original value :  {num_housing['median_income'].iloc[0]}") # 첫 번째 데이터의 실제 소득 값
    print(f"Standardize result : {housing_std[0,7]:.4f}" ) # StandardScaler 결과. 데이터를 평균 기준으로 얼마나 떨어져 있는지 , [0,7] 의 경우 첫 번째 데이터의 median_income 값
    print(f"Normalized result : {housing_minmax[0,7]:.4f}")# MinMaxScaler 결과. 전체 데이터 중 83% 위치

    print("\n Step 6: 정규화 이후 Min/Max 값")
    print("-" * 50)

    print(f"Min value (All feature) : {housing_minmax.min()}")
    print(f"Max value (All feature) : {housing_minmax.max()}")

