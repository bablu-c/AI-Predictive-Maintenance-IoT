from sklearn.model_selection import train_test_split

def preprocess_data(data):
    # Remove missing values
    data = data.dropna()

    # Features and target
    X = data.drop("failure", axis=1)
    y = data["failure"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test