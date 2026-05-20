import numpy as np

def generate_markov_noise(y: np.ndarray, eta: float, alpha_markov: float, random_state: int) -> np.ndarray:
    """
    Generate label noise according to a Markov chain.
    The noise process is an indicator sequence e_t in {0, 1}.
    With base noise rate eta and Markov transition probability alpha_markov.
    
    alpha_markov is interpreted here as the persistence of the noise state.
    P(e_t = 1 | e_{t-1} = 1) = alpha_markov
    P(e_t = 1) = eta
    
    This implies:
    P(e_t = 1 | e_{t-1} = 0) * (1 - eta) + P(e_t = 1 | e_{t-1} = 1) * eta = eta
    P(e_t = 1 | e_{t-1} = 0) * (1 - eta) + alpha_markov * eta = eta
    P(e_t = 1 | e_{t-1} = 0) = eta * (1 - alpha_markov) / (1 - eta)
    
    Constraints for exactly valid Markov chains:
    alpha_markov in [0, 1]
    eta * (1 - alpha_markov) / (1 - eta) in [0, 1]
    
    If eta=0, no noise.
    """
    if eta == 0.0:
        return y.copy()

    rng = np.random.default_rng(random_state)
    T = len(y)
    e = np.zeros(T, dtype=int)
    
    # Calculate transition probabilities
    p11 = alpha_markov
    p01 = eta * (1 - alpha_markov) / (1 - eta) if eta != 1.0 else 0.0
    
    # Clip prob to [0,1] for edge cases gracefully
    p01 = np.clip(p01, 0.0, 1.0)
    p11 = np.clip(p11, 0.0, 1.0)

    # Initial state
    e[0] = 1 if rng.random() < eta else 0
    
    for t in range(1, T):
        if e[t-1] == 1:
            e[t] = 1 if rng.random() < p11 else 0
        else:
            e[t] = 1 if rng.random() < p01 else 0

    y_noisy = y.copy()
    y_noisy[e == 1] = -y_noisy[e == 1]
    return y_noisy

def compute_bayes_risk(eta: float) -> float:
    """
    Computes the Bayes risk under uniform random noise or Markov noise.
    For symmetric label noise that flips y with probability eta, 
    the Bayes risk is simply eta assuming the underlying deterministic boundary is perfectly separable 
    and the clean accuracy formulation allows perfect performance.
    """
    return eta

def generate_iid_noise(y: np.ndarray, eta: float, random_state: int) -> np.ndarray:
    """
    Generate independent and identically distributed (IID) label noise.
    Flips each label independently with probability eta.
    """
    if eta == 0.0:
        return y.copy()
        
    rng = np.random.default_rng(random_state)
    T = len(y)
    
    # Generate flips: 1 with probability eta, 0 otherwise
    flips = rng.random(T) < eta
    
    y_noisy = y.copy()
    y_noisy[flips] = -y_noisy[flips]
    return y_noisy
