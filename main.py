from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model

def main():
    data = load_data("data/sensor_data.csv")

    X_train, X_test, y_train, y_test = preprocess_data(data)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()