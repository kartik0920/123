# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
 from sklearn.model_selection import train_test_split
 # Assign test data size 20%
 xtrain, xtest, ytrain, ytest =train_test_split(X,Y,test_size= 0.25, random_state=0)
 # from sklearn.linear_model import LogisticRegression
 # model = LogisticRegression(solver = 'lbfgs')
 from sklearn.naive_bayes import GaussianNB
 model = GaussianNB()
 model = model.fit(xtrain,ytrain)
 ypred = model.predict(xtest)

 from sklearn.metrics import confusion_matrix
 cm = confusion_matrix(ytest, ypred)
 sns.heatmap(cm, annot=True)
 plt.show()
 
 accuracy = accuracy_score(ytest, ypred)
 precision = precision_score(ytest, ypred, average='macro')  # or 'weighted' / 'micro'
 recall = recall_score(ytest, ypred, average='macro')

 print("\nModel Evaluation:")
 print("Accuracy:", accuracy)
 print("Precision:", precision)
 print("Recall:", recall)
 
 from sklearn.metrics import classification_report
 print(classification_report(ytest, ypred))
#---------------------------------------------------------------------------------------
# Reading dataset
df = sns.load_dataset('iris')

df.columns = ('SL', 'SW', 'PL', 'PW', 'Species')
# Display basic information
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
#---------------------------------------------------------------------------------------
# Display Statistical information
print('Statistical information of Numerical Columns: \n',df.describe())
#---------------------------------------------------------------------------------------
# Display Null values
print('Total Number of Null Values in Dataset:', df.isna().sum())
#---------------------------------------------------------------------------------------
# Label encoding method
df['Species']=df['Species'].astype('category')
df['Species']=df['Species'].cat.codes

df[['SL','SW', 'PL', 'PW']]=df[['SL','SW', 'PL', 'PW']].fillna(df[['SL','SW', 'PL', 'PW']].mean())

# Display correlation matrix
sns.heatmap(df.corr(),annot=True)
plt.show()
#---------------------------------------------------------------------------------------
# Choosing input and output variables from correlation matrix
X = df[['SL','SW', 'PL', 'PW']]
Y = df['Species']
BuildModel(X, Y)
#---------------------------------------------------------------------------------------
# Checking model score after removing outliers
fig, axes = plt.subplots(2,2)
sns.boxplot(data = df, x ='SL', ax=axes[0,0])
sns.boxplot(data = df, x ='SW', ax=axes[0,1])
sns.boxplot(data = df, x ='PL', ax=axes[1,0])
sns.boxplot(data = df, x ='PW', ax=axes[1,1])
plt.show()
df = RemoveOutlier(df, 'SW')
# Choosing input and output variables from correlation matrix
X = df[['SL','SW', 'PL', 'PW']]
Y = df['Species']
BuildModel(X, Y)
#After removing outliers accuracy is reducing due to overfitting of the model


#A 3x3 confusion matrix will look like this:

#[𝑇𝑃Setosa 𝐹𝑃Versicolor 𝐹𝑃Virginica
#𝐹𝑁Setosa 𝑇𝑃Versicolor 𝐹𝑃Virginica
#𝐹𝑁Setosa 𝐹𝑁Versicolor 𝑇𝑃Virginica]
​
  

 
​
  
​
