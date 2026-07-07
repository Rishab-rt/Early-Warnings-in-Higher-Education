import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


df = pd.read_csv('data/data.csv', sep=';')
print("Dataset loaded successfully!")
print(f"Dataset Shape: {df.shape}\n")


print("First 5 rows:")
print(df.head())

print("\nTarget Variable Distribution:")
print(df['Target'].value_counts())