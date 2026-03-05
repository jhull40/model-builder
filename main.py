import pandas as pd
from sklearn.datasets import load_iris
from model_builder.utils.utils import load_config
from model_builder import DataAnalyzer

if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["target"] = y

    config = load_config("configs/config.yaml")
    print("Config loaded:", config)

    analyzer = DataAnalyzer(config)
    analyzer.run(df)
    print("EDA complete")
