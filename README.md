# Cyberbullying-detection

This tool is a simple application designed to detect cyberbullying in messages or tweets using a machine learning model. It processes the input text, predicts whether it's an instance of cyberbullying, and tells the user the type of bullying. Here's how it works, step by step, explained in a way that's easy to understand:

What Does It Do?
Purpose: Helps identify if a message is an example of cyberbullying.
How It Works:
It takes a message (like a tweet) as input.
Uses a pre-trained model to classify the message.
Outputs the type of cyberbullying (e.g., hate speech, harassment, etc.).
How the Tool Works (Behind the Scenes)
Step 1: The Data
Dataset: It uses a file called cyberbullying_tweets.csv containing tweets labeled with the type of cyberbullying (or "not cyberbullying").
Example:
Tweet: "You're so stupid, why even try?"
Label: "harassment".
Step 2: Train the Model
Text Preprocessing:

Text (the tweet) is converted into numbers using a method called vectorization.
This method turns words into a format that the machine can understand.
Common, less important words (like "the" and "and") are ignored.
Learning Patterns:

A Naive Bayes model is trained using this numerical data.
It learns patterns in the text that distinguish between different types of cyberbullying.
Saving the Model:

Once trained, the model is saved into a .pkl file, making it reusable.
Step 3: Building the GUI
User Interface:

A simple Graphical User Interface (GUI) is built using Tkinter, a Python library.
The interface has:
A text box to input the message.
A button to submit the message for classification.
A display area to show the result.
How It Works for the User:

The user types a message into the app.
The app processes the message using the saved model.
It predicts and displays the type of cyberbullying.
Why Is This Tool Useful?
Awareness: Helps detect harmful messages that could hurt someone emotionally.
Efficiency: Automatically classifies messages without manual effort.
Practical: Can be used by social media platforms, educators, or researchers to flag problematic content.
Using the Tool
Start the App:

The app opens a window where you can input a message.
Input Message:

Type the message you want to check (e.g., "You're worthless and shouldn't be here.").
Get Result:

The app will tell you the type of cyberbullying (e.g., hate speech).
Example Output:

Input: "You're worthless and shouldn't be here."
Output: "Hate Speech"
Summary for Beginners
What It Does: Detects whether a message is cyberbullying and identifies the type.
How It Works: Uses machine learning to classify messages based on patterns it learned from tweets.
How to Use It: Just type a message, and the app tells you if it's cyberbullying.
