from sklearn.datasets import load_iris
from model_builder.utils.utils import load_config
from model_builder import DataAnalyzer

if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target

    # Print the shape of the data
    print("Data shape:", X.shape)
    print("Target shape:", y.shape)

    config = load_config("configs/config.yaml")
    print("Config loaded:", config)

    analyzer = DataAnalyzer(config)
    print("DataAnalyzer initialized with config:", analyzer.config)
