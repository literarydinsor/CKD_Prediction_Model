#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install -U imbalanced-learn

# In[ ]:


!pip install tensorflow

# In[ ]:


!pip install keras

# In[ ]:


import numpy as np
import pandas as pd

# Set a seed for reproducibility
np.random.seed(0)

# Number of individuals
n = 10000

# Age
age = np.random.uniform(20, 80, n)

# Gender
gender = np.random.choice(['Male', 'Female'], n)

# BMI
bmi = np.random.normal(25, 5, n)

# In[ ]:


# Diabetes
diabetes = np.random.choice([0, 1], n, p=[0.907, 0.093])

# Hypertension
hypertension = np.random.choice([0, 1], n, p=[0.71, 0.29])

# Hyperlipidemia
hyperlipidemia = np.random.choice([0, 1], n, p=[0.67, 0.33])

# History of using NSAID
nsaid_use = np.random.choice([0, 1], n, p=[0.7, 0.3])

# History of using herbs and herbal products
herb_use = np.random.choice([0, 1], n, p=[0.9, 0.1])

# Family history of CKD
family_history = np.random.choice([0, 1], n, p=[0.95, 0.05])

# In[ ]:


# Average daily salt intake
salt_intake = np.random.normal(3.4, 1, n)

# Combine all variables into a DataFrame
df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'BMI': bmi,
    'Diabetes': diabetes,
    'Hypertension': hypertension,
    'Hyperlipidemia': hyperlipidemia,
    'NSAID Use': nsaid_use,
    'Herb Use': herb_use,
    'Family History of CKD': family_history,
    'Average Daily Salt Intake': salt_intake
})

# Display the first few rows of the DataFrame
df.head()

# In[ ]:


# Check for missing values
missing_values = df.isnull().sum()
print('Missing values:', missing_values)

# Get summary statistics
summary = df.describe()
print('\nSummary statistics:')
print(summary)

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the plots
sns.set_style('whitegrid')

# Create subplots
fig, axs = plt.subplots(3, 4, figsize=(20, 15))

# Plot histograms for continuous variables
continuous_vars = ['Age', 'BMI', 'Average Daily Salt Intake']
for i, var in enumerate(continuous_vars):
    sns.histplot(df[var], kde=True, ax=axs[i, 0])

# Plot bar plots for categorical variables
categorical_vars = ['Gender', 'Diabetes', 'Hypertension', 'Hyperlipidemia', 'NSAID Use', 'Herb Use', 'Family History of CKD']
for i, var in enumerate(categorical_vars):
    df[var].value_counts().plot(kind='bar', ax=axs[i//3, i%3+1])
    axs[i//3, i%3+1].set_title(var)

# Adjust the layout
fig.tight_layout()

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define the feature matrix X and the target variable y
X = df.drop('Family History of CKD', axis=1)
y = df['Family History of CKD']

# Convert categorical variables into dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize a Logistic Regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# In[ ]:


# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Create the 'less likely, likely, more likely, and most likely' classification
classification = pd.cut(y_pred_proba, bins=[0, 0.25, 0.5, 0.75, 1], labels=['Less Likely', 'Likely', 'More Likely', 'Most Likely'])

# Create a DataFrame with the test set predictions
predictions = pd.DataFrame({'Predicted CKD': y_pred, 'Probability of CKD': y_pred_proba, 'Classification': classification}, index=y_test.index)
predictions.head()

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Calculate the performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print the performance metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'ROC AUC Score: {roc_auc:.2f}')

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest model
model_rf = RandomForestClassifier(random_state=0)

# Train the model on the training set
model_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = model_rf.predict(X_test)
y_pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]

# Create the 'less likely, likely, more likely, and most likely' classification
classification_rf = pd.cut(y_pred_proba_rf, bins=[0, 0.25, 0.5, 0.75, 1], labels=['Less Likely', 'Likely', 'More Likely', 'Most Likely'])

# Create a DataFrame with the test set predictions
predictions_rf = pd.DataFrame({'Predicted CKD': y_pred_rf, 'Probability of CKD': y_pred_proba_rf, 'Classification': classification_rf}, index=y_test.index)
predictions_rf.head()

# In[ ]:


# Calculate the performance metrics for the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

# Print the performance metrics
print(f'Accuracy (Random Forest): {accuracy_rf:.2f}')
print(f'Precision (Random Forest): {precision_rf:.2f}')
print(f'Recall (Random Forest): {recall_rf:.2f}')
print(f'F1 Score (Random Forest): {f1_rf:.2f}')
print(f'ROC AUC Score (Random Forest): {roc_auc_rf:.2f}')

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

# Initialize a Gradient Boosting model
model_gb = GradientBoostingClassifier(random_state=0)

# Train the model on the training set
model_gb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_gb = model_gb.predict(X_test)
y_pred_proba_gb = model_gb.predict_proba(X_test)[:, 1]

# Create the 'less likely, likely, more likely, and most likely' classification
classification_gb = pd.cut(y_pred_proba_gb, bins=[0, 0.25, 0.5, 0.75, 1], labels=['Less Likely', 'Likely', 'More Likely', 'Most Likely'])

# Create a DataFrame with the test set predictions
predictions_gb = pd.DataFrame({'Predicted CKD': y_pred_gb, 'Probability of CKD': y_pred_proba_gb, 'Classification': classification_gb}, index=y_test.index)
predictions_gb.head()

# In[ ]:


# Calculate the performance metrics for the Gradient Boosting model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb)
recall_gb = recall_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
roc_auc_gb = roc_auc_score(y_test, y_pred_proba_gb)

# Print the performance metrics
print(f'Accuracy (Gradient Boosting): {accuracy_gb:.2f}')
print(f'Precision (Gradient Boosting): {precision_gb:.2f}')
print(f'Recall (Gradient Boosting): {recall_gb:.2f}')
print(f'F1 Score (Gradient Boosting): {f1_gb:.2f}')
print(f'ROC AUC Score (Gradient Boosting): {roc_auc_gb:.2f}')

# In[ ]:


from imblearn.over_sampling import SMOTE

# Initialize a SMOTE instance
smote = SMOTE(random_state=0)

# Oversample the minority class in the training set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the Gradient Boosting model on the resampled training set
model_gb_resampled = GradientBoostingClassifier(random_state=0)
model_gb_resampled.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_gb_resampled = model_gb_resampled.predict(X_test)
y_pred_proba_gb_resampled = model_gb_resampled.predict_proba(X_test)[:, 1]

# Create the 'less likely, likely, more likely, and most likely' classification
classification_gb_resampled = pd.cut(y_pred_proba_gb_resampled, bins=[0, 0.25, 0.5, 0.75, 1], labels=['Less Likely', 'Likely', 'More Likely', 'Most Likely'])

# Create a DataFrame with the test set predictions
predictions_gb_resampled = pd.DataFrame({'Predicted CKD': y_pred_gb_resampled, 'Probability of CKD': y_pred_proba_gb_resampled, 'Classification': classification_gb_resampled}, index=y_test.index)
predictions_gb_resampled.head()

# In[ ]:


from imblearn.over_sampling import SMOTE

# Initialize a SMOTE instance
smote = SMOTE(random_state=0)

# Oversample the minority class in the training set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the Gradient Boosting model on the resampled training set
model_gb_resampled = GradientBoostingClassifier(random_state=0)
model_gb_resampled.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_gb_resampled = model_gb_resampled.predict(X_test)
y_pred_proba_gb_resampled = model_gb_resampled.predict_proba(X_test)[:, 1]

# Create the 'less likely, likely, more likely, and most likely' classification
classification_gb_resampled = pd.cut(y_pred_proba_gb_resampled, bins=[0, 0.25, 0.5, 0.75, 1], labels=['Less Likely', 'Likely', 'More Likely', 'Most Likely'])

# Create a DataFrame with the test set predictions
predictions_gb_resampled = pd.DataFrame({'Predicted CKD': y_pred_gb_resampled, 'Probability of CKD': y_pred_proba_gb_resampled, 'Classification': classification_gb_resampled}, index=y_test.index)
predictions_gb_resampled.head()

# In[ ]:


# Calculate the performance metrics for the Gradient Boosting model trained on the oversampled training set
accuracy_gb_resampled = accuracy_score(y_test, y_pred_gb_resampled)
precision_gb_resampled = precision_score(y_test, y_pred_gb_resampled)
recall_gb_resampled = recall_score(y_test, y_pred_gb_resampled)
f1_gb_resampled = f1_score(y_test, y_pred_gb_resampled)
roc_auc_gb_resampled = roc_auc_score(y_test, y_pred_proba_gb_resampled)

# Print the performance metrics
print(f'Accuracy (Gradient Boosting, Resampled): {accuracy_gb_resampled:.2f}')
print(f'Precision (Gradient Boosting, Resampled): {precision_gb_resampled:.2f}')
print(f'Recall (Gradient Boosting, Resampled): {recall_gb_resampled:.2f}')
print(f'F1 Score (Gradient Boosting, Resampled): {f1_gb_resampled:.2f}')
print(f'ROC AUC Score (Gradient Boosting, Resampled): {roc_auc_gb_resampled:.2f}')

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler

# Initialize a RandomUnderSampler instance
rus = RandomUnderSampler(random_state=0)

# Undersample the majority class in the training set
X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train, y_train)

# Train the Gradient Boosting model on the undersampled training set
model_gb_undersampled = GradientBoostingClassifier(random_state=0)
model_gb_undersampled.fit(X_train_undersampled, y_train_undersampled)

# Make predictions on the test set
y_pred_gb_undersampled = model_gb_undersampled.predict(X_test)
y_pred_proba_gb_undersampled = model_gb_undersampled.predict_proba(X_test)[:, 1]

# Create the 'less likely, likely, more likely, and most likely' classification
classification_gb_undersampled = pd.cut(y_pred_proba_gb_undersampled, bins=[0, 0.25, 0.5, 0.75, 1], labels=['Less Likely', 'Likely', 'More Likely', 'Most Likely'])

# Create a DataFrame with the test set predictions
predictions_gb_undersampled = pd.DataFrame({'Predicted CKD': y_pred_gb_undersampled, 'Probability of CKD': y_pred_proba_gb_undersampled, 'Classification': classification_gb_undersampled}, index=y_test.index)
predictions_gb_undersampled.head()

# In[ ]:


# Calculate the performance metrics for the Gradient Boosting model trained on the undersampled training set
accuracy_gb_undersampled = accuracy_score(y_test, y_pred_gb_undersampled)
precision_gb_undersampled = precision_score(y_test, y_pred_gb_undersampled)
recall_gb_undersampled = recall_score(y_test, y_pred_gb_undersampled)
f1_gb_undersampled = f1_score(y_test, y_pred_gb_undersampled)
roc_auc_gb_undersampled = roc_auc_score(y_test, y_pred_proba_gb_undersampled)

# Print the performance metrics
print(f'Accuracy (Gradient Boosting, Undersampled): {accuracy_gb_undersampled:.2f}')
print(f'Precision (Gradient Boosting, Undersampled): {precision_gb_undersampled:.2f}')
print(f'Recall (Gradient Boosting, Undersampled): {recall_gb_undersampled:.2f}')
print(f'F1 Score (Gradient Boosting, Undersampled): {f1_gb_undersampled:.2f}')
print(f'ROC AUC Score (Gradient Boosting, Undersampled): {roc_auc_gb_undersampled:.2f}')

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

# Define the architecture of the Neural Network
model_nn = Sequential()
model_nn.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model_nn.add(Dense(8, activation='relu'))
model_nn.add(Dense(1, activation='sigmoid'))

# Compile the Neural Network
model_nn.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=['accuracy'])

# Train the Neural Network
model_nn.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

# Define the architecture of the Neural Network
model_nn = Sequential()
model_nn.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model_nn.add(Dense(8, activation='relu'))
model_nn.add(Dense(1, activation='sigmoid'))

# Compile the Neural Network
model_nn.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=['accuracy'])

# Train the Neural Network
model_nn.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

# Define the architecture of the Neural Network
model_nn = Sequential()
model_nn.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model_nn.add(Dense(8, activation='relu'))
model_nn.add(Dense(1, activation='sigmoid'))

# Compile the Neural Network
model_nn.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=['accuracy'])

# Train the Neural Network
model_nn.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

# In[ ]:


# Evaluate the model on the test data
loss, accuracy = model_nn.evaluate(X_test, y_test, verbose=1)
loss, accuracy

# In[ ]:


# Predict probabilities for the test set
y_pred_probs = model_nn.predict(X_test)

# Define thresholds for the categories
thresholds = [0.25, 0.5, 0.75]

# Define function to convert probabilities to categories
def prob_to_category(prob, thresholds):
    if prob < thresholds[0]:
        return 'Less likely'
    elif prob < thresholds[1]:
        return 'Likely'
    elif prob < thresholds[2]:
        return 'More likely'
    else:
        return 'Most likely'

# Convert probabilities to categories
y_pred_categories = [prob_to_category(prob, thresholds) for prob in y_pred_probs]

y_pred_categories[:10]  # Show the first 10 predictions

# In[ ]:


# Check the shape of the training data
X_train.shape

# In[ ]:


import numpy as np

# Simulate a new, unseen data point
# For simplicity, we'll just use the mean values of the features in the training set
new_data = np.array([X_train.mean(axis=0)])

# Use the model to predict the probability of developing CKD for this new data point
new_data_pred_prob = model_nn.predict(new_data)[0][0]

# Convert the probability to a category
new_data_pred_category = prob_to_category(new_data_pred_prob, thresholds)

new_data_pred_prob, new_data_pred_category

# In[ ]:


# Create a new, unseen data point
# This is a mutable array that you can modify later
new_data = np.array([[70,  # age
                      1,   # gender (1 for male, 0 for female)
                      30,  # BMI
                      1,   # having diabetes (1 for yes, 0 for no)
                      1,   # having hypertension (1 for yes, 0 for no)
                      0,   # having hyperlipidemia (1 for yes, 0 for no)
                      0,   # history of using NSAID (1 for yes, 0 for no)
                      1,   # history of having herbs and herbal products (1 for yes, 0 for no)
                      0   # family history of CKD (1 for yes, 0 for no)
                      #5000    # average daily salt intake (in grams)
                     ]])

# Print the new data point
new_data

# In[ ]:


# Use the model to predict the probability of developing CKD for this new data point
new_data_pred_prob = model_nn.predict(new_data)[0][0]

# Convert the probability to a category
new_data_pred_category = prob_to_category(new_data_pred_prob, thresholds)

new_data_pred_prob, new_data_pred_category

# In[ ]:


# Prepare the data
X = df.drop('CKD', axis=1)
y = df['CKD']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# In[ ]:


# Check the columns in the dataframe
df.columns

# In[ ]:


# Recreate the simulated data
np.random.seed(42)
n = 10000

# Features
age = np.random.normal(50, 10, n)
gender = np.random.choice([0, 1], n)
bmi = np.random.normal(25, 5, n)
diabetes = np.random.choice([0, 1], n, p=[0.9, 0.1])
hypertension = np.random.choice([0, 1], n, p=[0.8, 0.2])
hyperlipidemia = np.random.choice([0, 1], n, p=[0.85, 0.15])
nsaid_use = np.random.choice([0, 1], n, p=[0.8, 0.2])
herb_use = np.random.choice([0, 1], n, p=[0.95, 0.05])
family_history = np.random.choice([0, 1], n, p=[0.9, 0.1])
salt_intake = np.random.normal(5, 2, n)

# Target variable
ckd = np.random.choice([0, 1], n, p=[0.95, 0.05])

# Create a dataframe
df = pd.DataFrame({'Age': age, 'Gender': gender, 'BMI': bmi, 'Diabetes': diabetes, 'Hypertension': hypertension,
                   'Hyperlipidemia': hyperlipidemia, 'NSAID Use': nsaid_use, 'Herb Use': herb_use,
                   'Family History of CKD': family_history, 'Average Daily Salt Intake': salt_intake, 'CKD': ckd})

# Check the dataframe
df.head()

# In[ ]:


# Prepare the data
X = df.drop('CKD', axis=1)
y = df['CKD']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# In[ ]:


# Define the model architecture
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')

# In[ ]:


# Create a new, unseen data point
new_data = np.array([[75, 1, 30, 1, 1, 1, 0, 1, 1, 11]])

# Normalize the new data point
new_data = scaler.transform(new_data)

# Use the model to predict the probability of developing CKD
prediction = model.predict(new_data)

# Print the prediction
print(f'Probability of developing CKD in the next 5 years: {prediction[0][0]*100:.2f}%')
