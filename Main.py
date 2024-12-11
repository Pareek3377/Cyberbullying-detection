import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import tkinter as tk
from tkinter import messagebox

# Step 1: Load and preprocess data
data = pd.read_csv('cyberbullying_tweets.csv')
X = data['tweet_text']
y = data['cyberbullying_type']

# Text vectorization
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, stratify=y, random_state=42)

# Step 2: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model
joblib.dump((model, vectorizer), 'cyberbullying_model.pkl')

# Step 3: Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 4: Tkinter Interface
def classify_message():
    message = message_box.get("1.0", tk.END).strip()
    if not message:
        messagebox.showwarning("Input Error", "Please enter a message!")
        return
    model, vectorizer = joblib.load('cyberbullying_model.pkl')
    prediction = model.predict(vectorizer.transform([message]))
    bullying_type.set(prediction[0])

app = tk.Tk()
app.title("Cyberbullying Detector")

tk.Label(app, text="Enter Message:").pack()
message_box = tk.Text(app, height=5, width=40)
message_box.pack()

tk.Label(app, text="Predicted Bullying Type:").pack()
bullying_type = tk.StringVar()
tk.Entry(app, textvariable=bullying_type, state="readonly").pack()

tk.Button(app, text="Submit", command=classify_message).pack()

app.mainloop()
