import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

'''
파라미터 : 함수의 가중치를 조절 . 주로 컴퓨터에서 정리됨
하이퍼 파라미터 : epoch, learning_rate, batch_size 를 조절하여 y 값을 결정

'''

def load_dataset():
    baspath = Path(__file__).parent
    csv_path = baspath / "datasets" / "stocks.csv"

    return pd.read_csv(csv_path, encoding='cp949')


if __name__ == "__main__":
    stock = load_dataset()
    pd.set_option('display.max_columns', None)

    print("---housing data head---")
    print(stock.head())
    print("\n Step1: Checking for missing value")
    print(stock.isnull().sum())

    print("\n Step 4: 스케이링 전 원본 데이터 출력")
    print("-" * 50)

    num_stock = stock.drop(["No", "Stock Name"], axis=1)
    print(num_stock.describe())

    std_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()

    stock_std = std_scaler.fit_transform(num_stock)
    stock_minmax = min_max_scaler.fit_transform(num_stock)
    print("\n Step 5: 스케일링 이후 데이터 Valuation (100M KRW)")

    print("-" * 50)

    print(f"Original value :  {num_stock['Valuation (100M KRW)'].iloc[0]}") # 첫 번째 데이터의 실제 소득 값
    print(f"Standardize result : {stock_std[0,0]:.4f}" ) # StandardScaler 결과. 데이터를 평균 기준으로 얼마나 떨어져 있는지
    print(f"Normalized result : {stock_minmax[0,0]:.4f}")# MinMaxScaler 결과.

    print("\n Step 6: 정규화 이후 Min/Max 값")
    print("-" * 50)

    print(f"Min value (All feature) : {stock_minmax.min()}")
    print(f"Max value (All feature) : {stock_minmax.max()}")

# -----------------------------------------
    base_path = Path(__file__).parent
    output_dir = base_path / "datasets"

    train_valid_set, test_set = train_test_split(stock, test_size=0.2, random_state=42)
    train_set, valid_set = train_test_split(train_valid_set, test_size=0.25, random_state=42)

    print("\n Step 1: 데이터 분할 결과 (6:2:2)")
    print("-" * 50)
    print(f"Total dataset:      {len(stock)}rows")
    print(f"Training dataset:   {len(train_set)}rows")
    print(f"Validation dataset: {len(valid_set)}rows")
    print(f"Test set :          {len(test_set)}rows")

    train_set.to_csv(output_dir / "stock_train.csv", index=False)
    valid_set.to_csv(output_dir / "stock_valid.csv", index=False)
    test_set.to_csv(output_dir / "stock_test.csv", index=False)

    print("\n Step 2: 데이터 저장 성공")
    print("-" * 50)
    print(f"Files saved in {output_dir}")