Advertising_data
Logistic Regression Project
Description
This project implements a logistic regression model using a dataset. The goal is to predict the probability of a specific event occurring based on independent variables, using logistic regression techniques. The model was evaluated using metrics such as Accuracy, Precision, Recall, and F1-Score.

Features
Data loading and preparation.
Logistic regression model creation.
Model training with train-test split.
Model evaluation using performance metrics.
Graphical visualizations for result analysis.
Requirements
To run this project, you need to have installed:

Python 3.12.8
Libraries:
pandas
numpy
scikit-learn
scipy
seaborn
matplotlib
Installation
Clone the repository:

git clone https://github.com/your-username/your-repository.git
Create a virtual environment:

python -m venv venv
Activate the virtual environment:

On Windows:

venv\Scripts\activate
On Linux/macOS:

source venv/bin/activate
Install dependencies:

pip install -r requirements.txt
How to Use
Data Preparation: Load the dataset, adjust the variables, and separate dependent and independent variables.
Model Creation and Training: Use the LogisticRegression function from scikit-learn to create and train the model using the training dataset.
Model Evaluation: Assess model performance using metrics such as Accuracy, Precision, Recall, and F1-Score.
Visualization: Generate plots such as ROC curves, scatter plots, and other visualizations to facilitate result interpretation.
Examples
Here is an example of how to run the model:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming 'data' is the loaded DataFrame
X = data.drop('target', axis=1)  # Independent variables
y = data['target']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')
Contribution
Fork this repository.
Create a branch for your feature (git checkout -b feature/MyFeature).
Make the necessary changes and commit (git commit -am 'Adding new feature').
Push the changes to your fork (git push origin feature/MyFeature).
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.