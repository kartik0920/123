import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score,precision_score,recall_score
#---------------------------------------------------------------------------------------
def RemoveOutlier(df, var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    outliers = df[(df[var] < low) | (df[var] > high)]
    print(f"{var} - Outliers detected: {len(outliers)}")
    
    return df[(df[var] >= low) & (df[var] <= high)]
#---------------------------------------------------------------------------------------
def BuildModel(X, Y):
 # Training and testing data

 # Assign test data size 20%
 xtrain, xtest, ytrain, ytest =train_test_split(X,Y,test_size= 0.25, random_state=13)

 model = LogisticRegression(solver = 'lbfgs')
 model = model.fit(xtrain,ytrain)
 ypred = model.predict(xtest)


 cm = confusion_matrix(ytest, ypred)
 sns.heatmap(cm, annot=True)
 plt.show()
 accuracy = accuracy_score(ytest,  ypred)
 precision = precision_score(ytest, ypred)
 recall = recall_score(ytest, ypred)

 print("\nModel Evaluation:")
 print("Accuracy:", accuracy)
 print("Precision:", precision)
 print("Recall:", recall)
 from sklearn.metrics import classification_report
 print(classification_report(ytest, ypred))
#---------------------------------------------------------------------------------------
# Reading dataset
df = pd.read_csv(r"Social_Network_Ads.csv")
# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
df = df.drop('User ID', axis=1)
df.columns = ['Gender', 'Age', 'Salary', 'Purchased']
#---------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Label encoding method
df['Gender']=df['Gender'].astype('category')
df['Gender']=df['Gender'].cat.codes
# Display correlation matrix
sns.heatmap(df.corr(),annot=True)
plt.show()
#---------------------------------------------------------------------------------------
# Choosing input and output variables from correlation matrix
X = df[['Age','Salary']]
Y = df['Purchased']
BuildModel(X, Y)
# #---------------------------------------------------------------------------------------
# Checking model score after removing outliers
fig, axes = plt.subplots(1,2)
sns.boxplot(data = df, x ='Age', ax=axes[0])
sns.boxplot(data = df, x ='Salary', ax=axes[1])
fig.tight_layout()
plt.show()
df = RemoveOutlier(df, 'Age')
df = RemoveOutlier(df, 'Salary')
# You can use normalization method to improve the score
# salary -> high range
# age -> low range
# Normalization will smoothe both range salary and age
# Choosing input and output variables from correlation matrix
X = df[['Age','Salary']]
Y = df['Purchased']
BuildModel(X, Y)