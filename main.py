import pandas as pd
from sklearn.datasets import make_classification
from model_builder.utils.utils import load_config
from model_builder.pipeline import Pipeline

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=524,
    )
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y

    config = load_config("configs/config.yaml")
    print("Config loaded:", config)

    pipeline = Pipeline(config)
    pipeline.run(df)

    print(f"Model ID : {pipeline.model_id}")
    print(f"Test score: {pipeline.test_score:.4f}")
    print("Pipeline complete")
