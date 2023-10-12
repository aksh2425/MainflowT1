#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy import stats
from sklearn.impute import SimpleImputer
df = pd.read_csv('01.Data Cleaning and Preprocessing.csv')
print("Initial Dataset:")
print(df.head())


# In[4]:


# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# In[17]:


from scipy.stats import zscore
from scipy import stats
#filling the missing values with mean of the respective column
numerical=df.select_dtypes(include='number').columns
imputer=SimpleImputer(strategy='mean')
df[numerical] = imputer.fit_transform(df[numerical])

# Handle Duplicates
df = df.drop_duplicates()

# Handle Outliers using z-score
z_scores = np.abs(stats.zscore(df[numerical]))
outliers = (z_scores < 3).all(axis=1)  # Adjust the threshold as needed
df = df[outliers]
df


# In[12]:


#Data visualisation
for i in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df[i], kde=True)
    plt.title(f'Before Outlier Removal - {i}')

    plt.subplot(1, 2, 2)
    sns.histplot(df[i], kde=True)
    plt.title(f'After Outlier Removal - {i}')

    plt.show()

# Save the cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)


# In[ ]:




