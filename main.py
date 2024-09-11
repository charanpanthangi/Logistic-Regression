# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load Dataset
# Replace 'your_dataset.csv' with the actual dataset path or URL
df = pd.read_csv('your_dataset.csv')

# Univariate Analysis (for numerical columns)
# This will plot the histogram of the chosen column to understand its distribution
df['your_column'].hist()
plt.title('Univariate Analysis')
plt.xlabel('Your Column')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis
# This will create a scatter plot to visualize the relationship between two variables
sns.scatterplot(x='your_independent_var', y='your_dependent_var', data=df)
plt.title('Bivariate Analysis')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable (Binary Outcome)')
plt.show()

# Split Dataset into Train and Test
# X is the independent variable(s), y is the dependent variable
X = df[['your_independent_var']]  # Independent variable(s)
y = df['your_dependent_var']      # Dependent variable (binary target: 0 or 1)

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
# Initialize the Logistic Regression model and train it using the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and Evaluate
# Predict the test data and calculate the accuracy score for evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
# Visualize the confusion matrix to evaluate the performance of the model
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# Save Model to .pkl file
# This saves the trained model as a pickle file for future use
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Loading the model and passing an input to the model
# Open the saved pickle file and load the model back
with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Test with a new input
# Example input: passing 5 as the independent variable value to the model
new_input = np.array([[5]])  # Example input
predicted_output = loaded_model.predict(new_input)
print(f'Predicted output (0 or 1) for input {new_input[0][0]}: {predicted_output[0]}')
