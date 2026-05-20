import numpy as np

def compute_plateau_average(accuracies: list[float], start_idx: int = 1000, end_idx: int = 1500) -> float:
    """
    Computes the plateau average of accuracies over the specified range.
    """
    if len(accuracies) <= start_idx:
        return float('nan') # Not enough rounds to reach the plateau start
    
    end = min(len(accuracies), end_idx)
    return float(np.mean(accuracies[start_idx:end]))

def identify_collapse(test_accuracies: list[float], window_size: int = 50, tolerance: float = 0.05) -> int:
    """
    Identifies the round at which AdaBoost "collapses", if any.
    Collapse is defined here formally:
    The test accuracy drops by at least `tolerance` below its maximum historical moving average, 
    and stays suppressed for at least `window_size` rounds.
    Returns the round index of the start of the collapse, or -1 if no collapse.
    """
    if len(test_accuracies) < window_size:
        return -1
        
    accuracies = np.array(test_accuracies)
    max_acc = -1.0
    
    for t in range(len(accuracies)):
        if accuracies[t] > max_acc:
            max_acc = accuracies[t]
            
        # Check if current accuracy is badly degraded
        if max_acc - accuracies[t] >= tolerance:
            # Check if it stays degraded
            if t + window_size <= len(accuracies):
                future_window = accuracies[t:t+window_size]
                if np.all(max_acc - future_window >= tolerance * 0.8): # sustained degradation
                    return t
    
    return -1

def compute_auc_normalized(learning_curve: list[float], bayes_accuracy: float) -> float:
    """
    Computes the Area Under the learning Curve (AUC), normalized by Bayes accuracy.
    bayes_accuracy = 1 - bayes_risk
    """
    # Simple rectangular or trapezoidal integration normalized by max ideal area
    auc = np.trapz(learning_curve)
    ideal_auc = bayes_accuracy * (len(learning_curve) - 1)
    
    if ideal_auc <= 0:
        return 0.0
        
    return float(auc / ideal_auc)
