# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split. 

4.Calculate Y_Pred and accuracy.

5.Print all the outputs. 

6.End the Program.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna()
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['label'].value_counts())
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vectorized, y_train)
y_pred = svm_model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "CONGRATULATIONS! You've won a FREE cruise to the Andaman! Call now to claim your prize!",
    "Can you pick up some milk on your way home?",
    "URGENT! Your account has been suspended. Click here to verify your details immediately."
]
print("\n" + "="*50)
print("TESTING WITH EXAMPLE MESSAGES:")
print("="*50)
for msg in test_messages:
    msg_vectorized = vectorizer.transform([msg])
    prediction = svm_model.predict(msg_vectorized)[0]
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"Message: {msg[:50]}... -> {result}")
```

## Output:

<img width="1913" height="965" alt="Screenshot 2026-03-19 113555" src="https://github.com/user-attachments/assets/3056b82e-f894-4686-aa5e-8ea08f95dca0" />


## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
