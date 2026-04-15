import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    print("Data Loaded Successfully")
    return data