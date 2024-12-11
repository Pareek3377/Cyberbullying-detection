# Cyberbullying-detection
This program is designed to build a simple cyberbullying detection system using a machine learning model trained on a dataset of tweets. It also includes a Graphical User Interface (GUI) to classify user-input messages. Let's break it down step by step:
Step 1: Load and Preprocess the Data
Load Data:

python
Copy code
data = pd.read_csv('cyberbullying_tweets.csv')
X = data['tweet_text']
y = data['cyberbullying_type']
The program reads a dataset named cyberbullying_tweets.csv.
The tweet_text column contains the tweet messages, and cyberbullying_type labels the type of bullying (e.g., "hate speech", "harassment").
Vectorize Text:

python
Copy code
vectorizer = CountVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
Converts text into numerical data that the machine learning model can understand.
Stop words (common words like "the", "and") are ignored to focus on important terms.
Train-Test Split:

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, stratify=y, random_state=42)
The data is split into training (70%) and testing (30%) sets.
stratify=y ensures the distribution of labels is consistent between training and testing sets.
Step 2: Train the Model
Model Training:

python
Copy code
model = MultinomialNB()
model.fit(X_train, y_train)
A Multinomial Naive Bayes classifier is used, which is effective for text classification.
The model learns patterns from the training data.
Save the Model:

python
Copy code
joblib.dump((model, vectorizer), 'cyberbullying_model.pkl')
Saves the trained model and vectorizer to a file for later use.
Step 3: Evaluate the Model
Make Predictions:

python
Copy code
y_pred = model.predict(X_test)
The model predicts the bullying type for the test data.
Evaluate Performance:

python
Copy code
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
Accuracy shows how often the model's predictions are correct.
Classification Report provides precision, recall, and F1 scores for each label.
Confusion Matrix shows where the model made correct and incorrect predictions.
Step 4: Create the GUI
Setup the Interface:

python
Copy code
app = tk.Tk()
app.title("Cyberbullying Detector")
Tkinter is used to build the GUI.
A window is created with the title "Cyberbullying Detector".
Add Widgets:

Input Box:

python
Copy code
tk.Label(app, text="Enter Message:").pack()
message_box = tk.Text(app, height=5, width=40)
message_box.pack()
A text box where the user enters a message to classify.
Output Box:

python
Copy code
tk.Label(app, text="Predicted Bullying Type:").pack()
bullying_type = tk.StringVar()
tk.Entry(app, textvariable=bullying_type, state="readonly").pack()
Displays the predicted bullying type based on the input.
Submit Button:

python
Copy code
tk.Button(app, text="Submit", command=classify_message).pack()
When clicked, it calls the classify_message function.
Define Functionality:

python
Copy code
def classify_message():
    message = message_box.get("1.0", tk.END).strip()
    if not message:
        messagebox.showwarning("Input Error", "Please enter a message!")
        return
    model, vectorizer = joblib.load('cyberbullying_model.pkl')
    prediction = model.predict(vectorizer.transform([message]))
    bullying_type.set(prediction[0])
Gets the input message from the text box.
Checks if the input is empty; if yes, shows a warning.
Loads the saved model and vectorizer.
Classifies the input message and updates the output box.
Run the App:

python
Copy code
app.mainloop()
Keeps the application running until the user closes it.
How It Works:
The program trains a machine learning model to classify tweets into cyberbullying types.
The GUI lets users input a message, processes it with the model, and displays the predicted bullying type.
You can try it by running the code, inputting a tweet, and seeing the result!
