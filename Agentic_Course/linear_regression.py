# ============================================================================
# 1. LINEAR REGRESSION (Review - You already know this!)
# ============================================================================
print("\n" + "=" * 80)
print("1. LINEAR REGRESSION - Predicting Continuous Values")
print("=" * 80)

# Data: House size (sq ft) vs Price ($1000s)
X_linear = np.array([600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]).reshape(-1, 1)
y_linear = np.array([150, 180, 220, 250, 280, 320, 350, 380, 410, 450])

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)

model_lr = LinearRegression()
model_lr.fit(X_train_lr, y_train_lr)
y_pred_lr = model_lr.predict(X_test_lr)

print(f"Slope: {model_lr.coef_[0]:.4f} (price increase per sq ft)")
print(f"Intercept: {model_lr.intercept_:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test_lr, y_pred_lr):.2f}")
print(f"\nExample: A 1500 sq ft house costs: ${model_lr.predict([[1500]])[0]:.2f}k")