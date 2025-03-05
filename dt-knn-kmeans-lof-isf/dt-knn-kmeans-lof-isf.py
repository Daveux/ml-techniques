#!/usr/bin/env python
# coding: utf-8

# ### **Step 1: Load the Dataset**

# In[55]:


import pandas as pd

# Load the dataset
df = pd.read_csv('titanic_train.csv')
print(f"Number of instances: {df.shape[0]}, Number of attributes: {df.shape[1]}")
df.head()


# ### **Step 2: Check for Data Quality Issues**

# In[61]:


df.info()
# Check for duplicate rows
duplicates = df[df.duplicated()]

# Display the number of duplicate rows
print(f"\nNumber of duplicate rows: {duplicates.shape[0]}")
# Display the number of missing values
print("\nMissing values per column:\n", df.isnull().sum())
df.describe()


# ### **Step 3: Remove Irrelevant Columns and convert datatypes**

# In[27]:


df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# Ensure appropriate data types
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Age'] = df['Age'].astype(float)  # Keep as float for potential NaN values
df['SibSp'] = df['SibSp'].astype(int)
df['Parch'] = df['Parch'].astype(int)
df['Fare'] = df['Fare'].astype(float)
df['Embarked'] = df['Embarked'].astype('category')

# Checking to confirm the changes
print(df.dtypes)


# ### **Step 4: Create New Columns**

# In[29]:


# AgeGroup
df['AgeGroup'] = pd.cut(df['Age'], bins=[-1, 0, 16, 30, 65, float('inf')],
                        labels=['NK', 'Child', 'Youth', 'Adult', 'Senior'], right=False)
df['AgeGroup'] = df['AgeGroup'].fillna('NK')  # Handling missing values

# Relatives
df['Relatives'] = df['SibSp'] + df['Parch']
df['Relatives'] = pd.cut(df['Relatives'], bins=[-1, 0, 3, float('inf')],
                         labels=['None', 'Few', 'Many'], right=False)
# Fare category
df['FareCategory'] = pd.cut(df['Fare'], bins=[-1, 0, 50, 100, float('inf')],
                            labels=['Free', 'Low', 'Average', 'High'], right=False)
df = df.drop(['SibSp', 'Parch', 'Age', 'Fare'], axis=1)
df


# ### **Step 5: One-Hot Encoding for Categorical Data**

# In[31]:


df = pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked', 'AgeGroup', 'Relatives', 'FareCategory'])
df


# ### **Step 6: Split Data into Train and Test Sets**

# In[33]:


from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df


# ### **Step 7: Train Decision Tree**

# In[35]:


from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)


# ### **Step 8: Predict on Test Set**

# In[37]:


y_pred = dt_model.predict(X_test)


# ### **Step 9: Evaluate Model**

# In[39]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot Decision Tree
plt.figure(figsize=(12,8))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'], fontsize=8)
plt.show()


# ### **Step 10: Prepare Data for Distance-Based Methods**

# In[41]:


df_distance = pd.read_csv('titanic_train.csv')
df_distance = df_distance.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df_distance['Relatives'] = df_distance['SibSp'] + df_distance['Parch']
df_distance = df_distance.drop(['SibSp', 'Parch'], axis=1)
# Calculate the median of the 'Age' column, ignoring NaN values
age_median = df_distance['Age'].median()
# Replace missing values in the 'Age' column with the median
#df_distance['Age'].fillna(age_median, inplace=True)
df_distance.fillna({'Age': age_median}, inplace=True)
df_distance


# ### **Step 11: Normalize Numerical Columns**

# In[43]:


from sklearn.preprocessing import normalize

df_distance[['Fare', 'Age', 'Relatives']] = normalize(df_distance[['Fare', 'Age', 'Relatives']], axis=0)
df_distance


# ### **Step 12: Encode Categorical Columns**

# In[45]:


df_distance = pd.get_dummies(df_distance, columns=['Sex', 'Pclass', 'Embarked'])
df_distance


# ### **Step 13: Classification with kNN**

# In[47]:


from sklearn.neighbors import KNeighborsClassifier
X_dist = df_distance.drop('Survived', axis=1)
y_dist = df_distance['Survived']
X_train_dist, X_test_dist, y_train_dist, y_test_dist = train_test_split(X_dist, y_dist, test_size=0.2, random_state=42)
df_distance
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_dist, y_train_dist)
y_pred_knn = knn_model.predict(X_test_dist)

print("kNN Accuracy:", accuracy_score(y_test_dist, y_pred_knn))
print("kNN Confusion Matrix:\n", confusion_matrix(y_test_dist, y_pred_knn))


# ### **Step 14: Clustering with kMeans**

# In[49]:


from sklearn.cluster import KMeans
import os
os.environ['OMP_NUM_THREADS'] = '3'

inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_dist)
    inertia.append(kmeans.inertia_)

# Plot elbow graph
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()


# ### **Step 15: Outlier Detection with LOF and ISF**

# In[51]:


from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

# LOF
lof = LocalOutlierFactor()
outliers_lof = lof.fit_predict(X_train_dist)

# ISF
isf = IsolationForest(random_state=42)
outliers_isf = isf.fit_predict(X_train_dist)

# Common outliers
common_outliers = (outliers_lof == -1) & (outliers_isf == -1)
print("Number of common outliers:", sum(common_outliers))


# In[ ]:




