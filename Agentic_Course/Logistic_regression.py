# ============================================================================
# 4. LOGISTIC REGRESSION - Classification (Binary)
# ============================================================================
print("\n" + "=" * 80)
print("4. LOGISTIC REGRESSION - Yes/No Classification")
print("=" * 80)

# Data: Student study hours vs Pass/Fail
# Features: [Study Hours, Previous Score]
X_log = np.array([
    [1, 45], [2, 50], [3, 55], [4, 60], [5, 65],
    [6, 70], [7, 75], [8, 80], [9, 85], [10, 90],
    [2, 40], [3, 48], [4, 52], [5, 58], [6, 62]
])
y_log = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])  # 0=Fail, 1=Pass

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.3, random_state=42)

model_log = LogisticRegression()
model_log.fit(X_train_log, y_train_log)
y_pred_log = model_log.predict(X_test_log)

print(f"Accuracy: {accuracy_score(y_test_log, y_pred_log) * 100:.1f}%")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test_log, y_pred_log))

# Probability prediction
new_student = [[5, 62]]
prob = model_log.predict_proba(new_student)[0]
print(f"\nStudent with 5 hours study, 62 previous score:")
print(f"  Probability of Failing: {prob[0]*100:.1f}%")
print(f"  Probability of Passing: {prob[1]*100:.1f}%")