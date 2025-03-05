#!/usr/bin/env python
# coding: utf-8

# ### **Step 1: Import Necessary Libraries and Load the Dataset**

# In[6]:


# Importing necessary libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()


# ### **Step 2: Print the number of samples in the dataset**

# In[9]:


print(f"Number of samples in the dataset: {digits.data.shape[0]}")


# ### **Step 3: Split data into train and test sets**

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=81)


# ### **Step 4: Create the MLP classifier**

# In[20]:


mlp = MLPClassifier(random_state=81)


# ### **Step 5: Fit the classifier on the training data**

# In[22]:


mlp.fit(X_train, y_train)


# ### **Step 6: Predict the digit for the test set.**

# In[24]:


y_pred = mlp.predict(X_test)


# ### **Step 7: Print accuracy, confusion matrix, and the classification report.**

# In[26]:


# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ### **Step 8: Print results for all samples in the test set**

# In[28]:


# Print results for all samples in the test set
for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
    print(f"Index: {i}, Actual: {actual}, Predicted: {predicted}")


# ### **Step 9: Plot the 81st image**

# In[30]:


# Plot the 81st image
index_81 = 80  # Index for the 81st test sample
plt.imshow(X_test[index_81].reshape(8, 8), cmap='gray')
plt.title(f"Actual: {y_test[index_81]}, Predicted: {y_pred[index_81]}")
plt.axis('off')
plt.show()

