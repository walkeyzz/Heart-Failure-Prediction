import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read Dataset
df = pd.read_excel(r"cancer.xlsx")

# Separate features (independent variables) and targets (dependent variables)
X = df[['Genetic Risk', 'Smoking']]
y = df['Level']

# Divide the data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create objects for KNN and Linear Regression
knn_model = KNeighborsClassifier()
regression_model = LinearRegression()

# Using the KNN algorithm
knn_model.fit(X_train, y_train)
knn_accuracy = knn_model.score(X_test, y_test)
print(f"KNN Accuracy: {knn_accuracy}")

# Using the Linear Regression algorithm
regression_model.fit(X_train[['Genetic Risk']], y_train)
regression_accuracy = regression_model.score(X_test[['Genetic Risk']], y_test)
print(f"Regression Accuracy: {regression_accuracy}")

# Make predictions on test data with KNN
y_pred_knn = knn_model.predict(X_test)

# Plot of KNN prediction results
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Genetic Risk'], X_test['Smoking'], c=y_pred_knn, cmap='viridis')
plt.xlabel('Genetic Risk')
plt.ylabel('Smoking')
plt.title('KNN Predictions')
plt.colorbar(label='Level')
plt.show()