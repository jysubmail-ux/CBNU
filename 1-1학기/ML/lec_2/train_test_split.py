import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

'''
파라미터 : 함수의 가중치를 조절 . 주로 컴퓨터에서 정리됨
하이퍼 파라미터 : epoch, learning_rate, batch_size 를 조절하여 y 값을 결정

'''

def load_dataset():
    baspath = Path(__file__).parent
    csv_path = baspath / "datasets" / "housing.csv"

    return pd.read_csv(csv_path)


if __name__ == "__main__":
    housing = load_dataset()
    pd.set_option('display.max_columns', None)

    base_path = Path(__file__).parent
    output_dir = base_path / "datasets"

    train_valid_set , test_set = train_test_split(housing, test_size=0.2, random_state=42)
    train_set, valid_set = train_test_split(train_valid_set, test_size=0.25, random_state=42)

    print("\n Step 1: 데이터 분할 결과 (6:2:2)")
    print("-" * 50)
    print(f"Total dataset:      {len(housing)}rows")
    print(f"Training dataset:   {len(train_set)}rows")
    print(f"Validation dataset: {len(valid_set)}rows")
    print(f"Test set :          {len(test_set)}rows")

    train_set.to_csv(output_dir / "housing_train.csv", index=False)
    valid_set.to_csv(output_dir / "housing_valid.csv", index=False)
    test_set.to_csv(output_dir / "housing_test.csv", index=False)

    print("\n Step 2: 데이터 저장 성공")
    print("-" * 50)
    print(f"Files saved in {output_dir}")