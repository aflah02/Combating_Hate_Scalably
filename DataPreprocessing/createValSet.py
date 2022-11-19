import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    df_train = pd.read_csv("../Data/PreprocessedData/train_preprocessed.csv")
    # Train Validation Split
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    df_train.to_csv("../Data/PreprocessedData/train_preprocessed.csv", index=False)
    df_val.to_csv("../Data/PreprocessedData/val_preprocessed.csv", index=False)