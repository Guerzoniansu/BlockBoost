import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.dgp.ar_process import generate_ar_process, generate_labels

def main():
    print("Testing Decision Stump (max_depth=1) on Clean Data...")
    
    n_samples = 3000
    n_features = 10
    ar_coeffs = [0.95] # AR(1) with rho=0.95
    noise_std = 1.0
    seed = 42

    print(f"Generating data: n_samples={n_samples}, n_features={n_features}, ar_coeffs={ar_coeffs}")
    X = generate_ar_process(
        n_samples=n_samples,
        n_features=n_features,
        ar_coeffs=ar_coeffs,
        noise_std=noise_std,
        random_state=seed
    )
    
    y_clean = generate_labels(X, random_state=seed)

    stump = DecisionTreeClassifier(max_depth=1)
    stump.fit(X, y_clean)
    
    y_pred = stump.predict(X)
    acc = accuracy_score(y_clean, y_pred)
    
    print(f"\nResult:")
    print(f"Decision Stump (clean train) accuracy: {acc:.4f}")
    
    if acc > 0.95:
        print("[+] Success: The stump achieves > 95% accuracy, indicating the dominant feature DGP works as expected.")
    else:
        print("[-] Warning: The stump accuracy is somewhat low. The DGP might need further tweaking if perfect accuracy was strictly expected.")

if __name__ == "__main__":
    main()
