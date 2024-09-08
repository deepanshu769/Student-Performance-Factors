import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv("StudentPerformanceFactors.csv")

# Drop rows with missing values
dataset = dataset.dropna()

# Print the summary statistics of the dataset
print(dataset.describe().T)

# Print the count of null values in each column (if any)
print(dataset.isnull().sum())

for column in dataset:
    print(dataset[column].value_counts()
          )
    print('-------------------')

for column in dataset:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataset, x=column, bins=20, kde=True)
    plt.title(f'Histogram of {column}')
    sns.countplot(x=column, data=dataset)
    plt.show()

sns.boxplot(x='Distance_from_Home',y='Attendance',data=dataset)
plt.figure(figsize=(8,8))
sns.lmplot(x='Hours_Studied', y='Exam_Score', data=dataset)
plt.show()

sns.boxplot(x='Parental_Involvement',y='Exam_Score', data=dataset)
plt.title('Parental Involement Vs Exam Score')
plt.show()

sns.violinplot(x='Family_Income',y='Exam_Score',data=dataset)
plt.figure(figsize=(10, 6))
sns.histplot(data=dataset, x=column, bins=20, kde=True)
plt.title('Family Income Vs Exam Score')
plt.show()

column=['Hours_Studied', 'Attendance', 
        'Previous_Scores', 'Sleep_Hours', 
        'Tutoring_Sessions']
sns.pairplot(dataset[column])
plt.suptitle('Pairplot of the Dataset', y=1.02)
plt.show()

sns.boxplot(x='Gender', y='Exam_Score', data=dataset)
plt.title('Gender Vs Exam Score')
plt.show()

sns.violinplot(x='Parental_Education_Level',y='Exam_Score',data=dataset
               )
plt.title('Parental Education Vs Exam Score')
plt.show()

sns.catplot(x='Sleep_Hours', y='Exam_Score', data=dataset, kind='box')
plt.title('Sleep Hours Vs Exam Score')
plt.show()


from sklearn.preprocessing import LabelEncoder
categerical_colum=['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
                       'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                       'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                       'Parental_Education_Level', 'Distance_from_Home', 'Gender']
Label_encoder={}
for column in categerical_colum:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])
    Label_encoder[column]=le

plt.figure(figsize=(10, 6))
correclation_matrix=dataset.corr()
sns.heatmap(correclation_matrix, annot=True, cmap='coolwarm',linewidths=0.6)
plt.title('Correlation Matrix')
plt.show()