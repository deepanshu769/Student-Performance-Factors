import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dataset ko import karna
dataset = pd.read_csv("StudentPerformanceFactors.csv")

# Missing values waale rows ko remove karna
dataset = dataset.dropna()

# Dataset ke summary statistics print karna
print(dataset.describe().T)

# Har column mein null values count karna (agar ho toh)
print(dataset.isnull().sum())

# Har column ke value counts ko print karna
for column in dataset:
    print(dataset[column].value_counts())
    print('-------------------')

# Har column ke histograms plot karna
for column in dataset:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataset, x=column, bins=20, kde=True)
    plt.title(f'Histogram of {column}')
    plt.show()

# Distance from Home vs Attendance ka boxplot
sns.boxplot(x='Distance_from_Home', y='Attendance', data=dataset)
plt.title('Distance from Home vs Attendance')
plt.show()

# Hours Studied vs Exam Score ka regression plot
sns.lmplot(x='Hours_Studied', y='Exam_Score', data=dataset)
plt.title('Hours Studied vs Exam Score')
plt.show()

# Parental Involvement vs Exam Score ka boxplot
sns.boxplot(x='Parental_Involvement', y='Exam_Score', data=dataset)
plt.title('Parental Involvement vs Exam Score')
plt.show()

# Family Income vs Exam Score ka violin plot
sns.violinplot(x='Family_Income', y='Exam_Score', data=dataset)
plt.title('Family Income vs Exam Score')
plt.show()

# Selected columns ka pairplot
columns = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions']
sns.pairplot(dataset[columns])
plt.suptitle('Pairplot of the Dataset', y=1.02)
plt.show()

# Gender vs Exam Score ka boxplot
sns.boxplot(x='Gender', y='Exam_Score', data=dataset)
plt.title('Gender vs Exam Score')
plt.show()

# Parental Education Level vs Exam Score ka violin plot
sns.violinplot(x='Parental_Education_Level', y='Exam_Score', data=dataset)
plt.title('Parental Education Level vs Exam Score')
plt.show()

# Sleep Hours vs Exam Score ka boxplot
sns.catplot(x='Sleep_Hours', y='Exam_Score', data=dataset, kind='box')
plt.title('Sleep Hours vs Exam Score')
plt.show()

# Categorical columns ko label encode karna
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
                       'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                       'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                       'Parental_Education_Level', 'Distance_from_Home', 'Gender']

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])
    label_encoders[column] = le

# Correlation matrix ko plot karna
plt.figure(figsize=(10, 6))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.6)
plt.title('Correlation Matrix')
plt.show()
