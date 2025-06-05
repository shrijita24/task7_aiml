import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset from CSV
df = pd.read_csv('breast-cancer.csv')  # Replace with your actual filename if different

# Preview to confirm columns
print(df.head())
print(df.columns)

# 2. Select two numeric features for visualization and 'target' as label
X = df[['radius_mean', 'texture_mean']]
  # Ensure these columns exist
y = df['diagnosis'].map({'M': 1, 'B': 0})  # Or use 'diagnosis' if the column is named that way (convert to binary if needed)

# 3. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train SVM with linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# 6. Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

# 7. Evaluate performance
print("Linear SVM Accuracy:", accuracy_score(y_test, svm_linear.predict(X_test)))
print("RBF SVM Accuracy:", accuracy_score(y_test, svm_rbf.predict(X_test)))

# 8. Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01]
}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

print("Best Params (RBF):", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# 9. Cross-validation score
cv_scores = cross_val_score(grid.best_estimator_, X_scaled, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

# 10. Plotting decision boundary
def plot_decision_boundary(model, X, y, title):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.astype(float).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.title(title)
    plt.xlabel('Mean Radius')
    plt.ylabel('Mean Texture')
    plt.grid(True)
    plt.show()

# 11. Visualize decision boundaries
plot_decision_boundary(svm_linear, X_scaled, y, 'SVM with Linear Kernel')
plot_decision_boundary(svm_rbf, X_scaled, y, 'SVM with RBF Kernel')
plot_decision_boundary(grid.best_estimator_, X_scaled, y, 'Tuned SVM with RBF Kernel')
