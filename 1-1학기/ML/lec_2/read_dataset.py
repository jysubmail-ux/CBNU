import pandas as pd
from pathlib import Path


#data = pd.read_csv('datasets/housing.csv')

def load_dataset():
    baspath = Path(__file__).parent
    csv_path = baspath / "datasets" / "housing.csv"

    return pd.read_csv(csv_path)



if __name__ == "__main__":

    housing = load_dataset()

    # print("---housing data head---")
    # print(housing.head())
    #
    # print("\n--- housing data info ---")
    # print(housing.info())

    print("\n Step1: Checking for missing value")
    print(housing.isnull().sum())

    print("\n Step2: Selecting Missing Data Strategy")

    # [Option 1] Drop rows with missing values
    # housing = housing.dropna(subset=["total_bedrooms"])
    # print("Result: Dropped rows containing missing values.")
    #
    # # [Option 2] Drop the entire column
    # housing = housing.drop("total_bedrooms", axis=1)
    # print("Result: Dropped rows containing missing values.")
    #
    # # [Option 3] inpute with median (Recommended)
    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)
    print(f"Result: Imputed missing values with median ({median}).")

    #Step 3: Final verification
    print("\n Step 3: Verification after processing")
    print(housing.isnull().sum())



