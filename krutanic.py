import pandas as pd
import numpy as np

# Example DataFrame
# customer_churn = pd.read_csv('your_data.csv')

# a. Extract the 5th column & store it in ‘customer_5’
customer_5 = customer_churn.iloc[:, 4]

# b. Extract the 15th column & store it in ‘customer_15’
customer_15 = customer_churn.iloc[:, 14]

# c. Extract all the male senior citizens whose Payment Method is Electronic check & store the result in ‘senior_male_electronic’
senior_male_electronic = customer_churn[
    (customer_churn['Gender'] == 'Male') &
    (customer_churn['SeniorCitizen'] == 1) &
    (customer_churn['PaymentMethod'] == 'Electronic check')
]

# d. Extract all those customers whose tenure is greater than 70 months or their Monthly charges is more than 100$ & store the result in ‘customer_total_tenure’
customer_total_tenure = customer_churn[
    (customer_churn['tenure'] > 70) |
    (customer_churn['MonthlyCharges'] > 100)
]

# e. Extract all the customers whose Contract is of two years, payment method is Mailed check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’
two_mail_yes = customer_churn[
    (customer_churn['Contract'] == 'Two year') &
    (customer_churn['PaymentMethod'] == 'Mailed check') &
    (customer_churn['Churn'] == 'Yes')
]

# f. Extract 333 random records from the customer_churn dataframe & store the result in ‘customer_333’
customer_333 = customer_churn.sample(n=333, random_state=1)

# g. Get the count of different levels from the ‘Churn’ column
churn_counts = customer_churn['Churn'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns

# a. Bar-plot for the ‘InternetService’ column
plt.figure(figsize=(10, 6))
sns.countplot(data=customer_churn, x='InternetService', color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of Categories')
plt.title('Distribution of Internet Service')
plt.show()

# b. Histogram for the ‘tenure’ column
plt.figure(figsize=(10, 6))
plt.hist(customer_churn['tenure'], bins=30, color='green')
plt.title('Distribution of tenure')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.show()

# c. Scatter-plot between ‘MonthlyCharges’ & ‘tenure’
plt.figure(figsize=(10, 6))
plt.scatter(customer_churn['tenure'], customer_churn['MonthlyCharges'], color='brown')
plt.xlabel('Tenure of customer')
plt.ylabel('Monthly Charges of customer')
plt.title('Tenure vs Monthly Charges')
plt.show()

# d. Box-plot between ‘tenure’ & ‘Contract’
plt.figure(figsize=(10, 6))
sns.boxplot(data=customer_churn, x='Contract', y='tenure')
plt.xlabel('Contract')
plt.ylabel('Tenure')
plt.title('Box Plot of Tenure by Contract')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# a. Build a simple linear model where dependent variable is ‘Monthly Charges’ and independent variable is ‘tenure’
X = customer_churn[['tenure']]
y = customer_churn['MonthlyCharges']

# i. Divide the dataset into train and test sets in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# ii. Build the model on train set and predict the values on test set
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# iii. Find the root mean square error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# iv. Find out the error in prediction & store the result in ‘error’
error = y_test - y_pred

# v. Find the root mean square error
print(f'Root Mean Square Error: {rmse}')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# a. Simple logistic regression model
X_simple = customer_churn[['MonthlyCharges']]
y_simple = customer_churn['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# i. Divide the dataset in 65:35 ratio
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.35, random_state=1)

# ii. Build the model on train set and predict the values on test set
log_model_simple = LogisticRegression()
log_model_simple.fit(X_train_simple, y_train_simple)
y_pred_simple = log_model_simple.predict(X_test_simple)

# iii. Build the confusion matrix and get the accuracy score
conf_matrix_simple = confusion_matrix(y_test_simple, y_pred_simple)
accuracy_simple = accuracy_score(y_test_simple, y_pred_simple)
print(f'Confusion Matrix (Simple):\n{conf_matrix_simple}')
print(f'Accuracy Score (Simple): {accuracy_simple}')

# b. Multiple logistic regression model
X_multiple = customer_churn[['tenure', 'MonthlyCharges']]
y_multiple = customer_churn['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# i. Divide the dataset in 80:20 ratio
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(X_multiple, y_multiple, test_size=0.2, random_state=1)

# ii. Build the model on train set and predict the values on test set
log_model_multiple = LogisticRegression()
log_model_multiple.fit(X_train_multiple, y_train_multiple)
y_pred_multiple = log_model_multiple.predict(X_test_multiple)

# iii. Build the confusion matrix and get the accuracy score
conf_matrix_multiple = confusion_matrix(y_test_multiple, y_pred_multiple)
accuracy_multiple = accuracy_score(y_test_multiple, y_pred_multiple)
print(f'Confusion Matrix (Multiple):\n{conf_matrix_multiple}')
print(f'Accuracy Score (Multiple): {accuracy_multiple}')
from sklearn.tree import DecisionTreeClassifier

# a. Decision tree model
X_decision = customer_churn[['tenure']]
y_decision = customer_churn['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# i. Divide the dataset in 80:20 ratio
X_train_decision, X_test_decision, y_train_decision, y_test_decision = train_test_split(X_decision, y_decision, test_size=0.2, random_state=1)

# ii. Build the model on train set and predict the values on test set
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_decision, y_train_decision)
y_pred_decision = tree_model.predict(X_test_decision)

# iii. Build the confusion matrix and calculate the accuracy
conf_matrix_decision = confusion_matrix(y_test_decision, y_pred_decision)
accuracy_decision = accuracy_score(y_test_decision, y_pred_decision)
print(f'Confusion Matrix (Decision Tree):\n{conf_matrix_decision}')
print(f'Accuracy Score (Decision Tree): {accuracy_decision}')
from sklearn.ensemble import RandomForestClassifier

# a. Random Forest model
X_forest = customer_churn[['tenure', 'MonthlyCharges']]
y_forest = customer_churn['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# i. Divide the dataset in 70:30 ratio
X_train_forest, X_test_forest, y_train_forest, y_test_forest = train_test_split(X_forest, y_forest, test_size=0.3, random_state=1)

# ii. Build the model on train set and predict the values on test set
forest_model = RandomForestClassifier()
forest_model.fit(X_train_forest, y_train_forest)
y_pred_forest = forest_model.predict(X_test_forest)

# iii. Build the confusion matrix and calculate the accuracy
conf_matrix_forest = confusion_matrix(y_test_forest, y_pred_forest)
accuracy_forest = accuracy_score(y_test_forest, y_pred_forest)
print(f'Confusion Matrix (Random Forest):\n{conf_matrix_forest}')
print(f'Accuracy Score (Random Forest): {accuracy_forest}')
