import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load():
    df = pd.read_csv('data/data.csv', sep=';')

    target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    df['Target'] = df['Target'].map(target_map)

    X = df.drop(columns=['Target'])
    y = df['Target']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n================ MODEL PERFORMANCE ================")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Dropout', 'Enrolled', 'Graduate']))
    print("===================================================\n")
    
    joblib.dump(model, 'logistic_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and Scaler successfully saved!")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load()
    train_regression(X_train, X_test, y_train, y_test)