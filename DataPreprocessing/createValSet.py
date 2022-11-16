import pandas as pd
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    folder_path = 'Data\PreprocessedData'
    df_train = pd.read_csv(os.path.join(folder_path, 'train_preprocessed.csv'))
    # Train Validation Split
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    df_train.to_csv(os.path.join(folder_path, 'train_preprocessed.csv'), index=False)
    df_val.to_csv(os.path.join(folder_path, 'val_preprocessed.csv'), index=False)