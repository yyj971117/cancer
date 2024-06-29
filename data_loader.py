import pandas as pd

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    return data
