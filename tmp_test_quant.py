import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Simple dataset
X = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 2, 3, 4, 10, 12, 14, 16, 18])

# We need tau = 0.95 (for left tail VaR, i.e. 95% quantile VaR)
tau = 0.95

model = GradientBoostingRegressor(loss='quantile', alpha=tau, n_estimators=1, learning_rate=1.0, max_depth=2, random_state=42)
model.fit(X, y)

preds = model.predict(X)
print("n_estimators=1 predictions:", preds)
print("initial estimate class:", type(model.init_))

model10 = GradientBoostingRegressor(loss='quantile', alpha=tau, n_estimators=10, learning_rate=0.1, max_depth=2, random_state=42)
model10.fit(X, y)
print("n_estimators=10 predictions:", model10.predict(X))
