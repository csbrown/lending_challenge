
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

import numpy as np


# In[3]:

import collections


# In[4]:

import sklearn, sklearn.tree


# In[5]:

df = pd.read_csv("data.csv")


# In[6]:

df["days_delinquent_old_bin"] = pd.cut(df["days_delinquent_old"], bins=[0,1,5,10,30,60,np.inf], include_lowest=True, right=False)


# In[7]:

df["days_delinquent_new_bin"] = pd.cut(df["days_delinquent_new"], bins=[0,1,5,10,30,60,np.inf], include_lowest=True, right=False)


# In[24]:

df


# In[8]:

def transition_matrix(series1, series2, weights=None):
    transition_matrix = pd.crosstab(series1, series2, values=weights, aggfunc=sum if weights is not None else None, normalize='index')
    return transition_matrix
    
def test_transition_matrix():
    d = pd.DataFrame([[1,2,3],[3,4,1],[1,4,1]])
    tm1 = transition_matrix(d[0],d[1])
    tm2 = transition_matrix(d[0],d[1],d[2])
    assert np.allclose(tm1.as_matrix(), np.array([[0.5,0.5],[0.0,1.0]]), rtol=0.001, atol=0)
    assert np.allclose(tm2.as_matrix(), np.array([[0.75,0.25],[0.0,1.0]]), rtol=0.001, atol=0)


# In[9]:

test_transition_matrix()


# In[10]:

df["gain"] = df["days_delinquent_new"] - df["days_delinquent_old"]


# In[11]:

df


# In[18]:

model = sklearn.tree.DecisionTreeRegressor(max_depth=5)
training_data = df[['average_bank_balance__c', 'new_outstanding_principal_balance', 'initial_loan_amount', 'fico', 'term', 'gain']].dropna()
model.fit(training_data[['average_bank_balance__c', 'new_outstanding_principal_balance', 'initial_loan_amount', 'fico', 'term']], training_data['gain'])


# In[19]:

model.feature_importances_


# In[20]:

sklearn.tree.export_graphviz(model, "tree.dot", )


# In[21]:

transition_matrix(df["days_delinquent_old_bin"], df["days_delinquent_new_bin"])


# In[22]:

transition_matrix(df["days_delinquent_old_bin"], df["days_delinquent_new_bin"], df["new_outstanding_principal_balance"])


# In[23]:

transition_matrix(df["days_delinquent_old_bin"], df["days_delinquent_new_bin"], df["fico"])


# In[ ]:




# In[ ]:



