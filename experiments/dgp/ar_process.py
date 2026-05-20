import numpy as np
from scipy import linalg
from typing import Tuple

def generate_ar_process(
    n_samples: int, 
    n_features: int, 
    ar_coeffs: list[float], 
    noise_std: float, 
    random_state: int
) -> np.ndarray:
    """
    Generates an AR(p) process of shape (n_samples, n_features).
    All features share the same AR coefficients but are generated independently 
    for simplicity, or we can treat the whole X as an independent multivariate generation.
    Here we generate n_features independent AR(p) series.

    X_t = c + phi_1 X_{t-1} + ... + phi_p X_{t-p} + e_t
    """
    rng = np.random.default_rng(random_state)
    p = len(ar_coeffs)
    phi = np.array(ar_coeffs)
    
    # Pre-allocate
    X = np.zeros((n_samples + p, n_features))
    
    # Generate innovations (Heavy-tailed Student-t with df=3)
    e = rng.standard_t(df=3, size=(n_samples + p, n_features)) * noise_std
    
    # AR generation
    for t in range(p, n_samples + p):
        # dot product of phi and the last p values of X.
        # X[t-1] is the most recent, X[t-p] is the oldest
        X[t] = sum(phi[i] * X[t - 1 - i] for i in range(p)) + e[t]
        
    return X[p:]

def generate_labels(X: np.ndarray, random_state: int) -> np.ndarray:
    """
    Generate clean labels based on the features.
    For simplicity, use a linear combination followed by a sign function.
    y_t = sign(w^T X_t)
    """
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    
    # Fixed true weights for the underlying classification boundary
    # Make 3 features dominate with comparable but distinct contributions
    w = rng.standard_normal(n_features) * 0.01 # Noise level for other features
    dominant_indices = rng.choice(n_features, size=3, replace=False)
    # Assign significant weights (e.g., 1.0, 0.7, 0.5) to the selected features
    weights_v = [1.0, 0.75, 0.5]
    for idx, weight in zip(dominant_indices, weights_v):
        w[idx] = weight * rng.choice([-1.0, 1.0])
    w /= np.linalg.norm(w)
    
    scores = X @ w
    # Map to {-1, 1}
    y = np.sign(scores)
    y[y == 0] = 1 # Handle exact 0
    return y

def compute_companion_matrix_spectral_radius(ar_coeffs: list[float]) -> float:
    """
    Computes the spectral radius of the companion matrix for an AR(p) process,
    which corresponds to the maximum magnitude eigenvalue.
    """
    p = len(ar_coeffs)
    if p == 0:
        return 0.0
        
    C = np.zeros((p, p))
    C[0, :] = ar_coeffs
    if p > 1:
        C[1:, :-1] = np.eye(p - 1)
        
    eigenvalues = linalg.eigvals(C)
    rho = np.max(np.abs(eigenvalues))
    return float(rho)

def compute_theta(ar_coeffs: list[float]) -> float:
    """
    Calculates the geometric mixing decay rate theta based on the companion matrix eigenvalues.
    theta = -ln(max|lambda|)
    If rho >= 1, the process is non-stationary and mixing is undefined (we return 0.0 or raise).
    """
    rho = compute_companion_matrix_spectral_radius(ar_coeffs)
    if rho >= 1.0 or rho == 0.0:
        return 1e-6 # fallback numerical stability
    return -np.log(rho)

def compute_theoretical_optimal_block_size(T: int, theta: float, eta: float) -> int:
    """
    Computes your exact theoretical optimal block size a_T* derived from the beta-mixing geometric decay rate theta.
    Formula: a_T* = (3 / (2 * theta)) * ln(T) - (1 / theta) * ln(1 - 2*eta)
    We round the result and bound it from below by 1.
    """
    term1 = (3.0 / (2.0 * theta)) * np.log(T)
    term2 = 0.0
    if eta < 0.5:
        term2 = (1.0 / theta) * np.log(1.0 - 2 * eta)
        
    a_T = term1 - term2
    return max(1, int(np.round(a_T)))
