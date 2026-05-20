import numpy as np
from model.blockboost import QBlockBoostR2Regressor
from sklearn.datasets import make_regression

def test_qblockboost():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_val, y_val = X[:20], y[:20]
    X_train, y_train = X[20:], y[20:]
    
    model = QBlockBoostR2Regressor(n_estimators=5, block_size=10)
    print("Fitting model...")
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    print("Fit successful.")
    
    print("Predicting...")
    preds = model.predict(X_train)
    print(f"Predictions shape: {preds.shape}")
    assert preds.shape == (80,)
    print("Verification successful!")

if __name__ == "__main__":
    test_qblockboost()
