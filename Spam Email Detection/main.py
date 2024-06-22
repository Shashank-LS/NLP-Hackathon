import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Text preprocessing function
def preprocess_text(text):
    """Preprocesses the input text for normalization."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Load and preprocess dataset
def load_and_preprocess_data():
    """Loads the dataset, selects relevant columns, and preprocesses the email text."""
    data = pd.read_csv('C:/Users/shash/OneDrive/Desktop/NLP Hackathon/spam.csv', encoding='latin1')
    data.columns = ['label', 'email', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    data = data[['label', 'email']]
    data['email'] = data['email'].apply(preprocess_text)
    return data

# Train the model
def train_model(data):
    """Trains a Naive Bayes classifier using the CountVectorizer on preprocessed text data."""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['email'])
    y = data['label'].map({'spam': 1, 'not spam': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam'])
    confusion = confusion_matrix(y_test, y_pred)

    return vectorizer, classifier, accuracy, report, confusion

# GUI Application
class SpamEmailDetectorApp:
    def __init__(self, root):
        """Initializes the GUI application with widgets and loads/trains the model."""
        self.root = root
        self.root.title("Spam Email Detector")

        # Set background color to black for root and all widgets
        self.root.configure(background='black')

        # Style configuration
        style = ttk.Style()
        style.configure('TFrame', background='black')
        style.configure('TLabel', background='black', foreground='white', font=('Arial', 12))
        style.configure('TButton', font=('Arial', 12, 'bold'))
        style.configure('Horizontal.TProgressbar', background='white')

        # Setup GUI widgets
        self.setup_widgets()

        # Load and train the model
        data = load_and_preprocess_data()
        self.vectorizer, self.classifier, self.accuracy, self.report, self.confusion = train_model(data)

        # Display model evaluation in GUI
        self.report_text.insert(tk.END, self.report)
        self.confusion_text.insert(tk.END, self.format_confusion_matrix(self.confusion))

        # Hide details initially
        self.report_label.grid_remove()
        self.report_text.grid_remove()
        self.confusion_label.grid_remove()
        self.confusion_text.grid_remove()

        # Bindings for real-time prediction
        self.email_text.bind('<KeyRelease>', self.predict_real_time)
        self.email_text.focus()

        # Bindings for keyboard shortcuts
        self.root.bind('<Control-Return>', lambda event: self.predict_spam(event))
        self.root.bind('<Control-BackSpace>', lambda event: self.clear_email_text(event))

    def setup_widgets(self):
        """Sets up all GUI widgets."""
        frame = ttk.Frame(self.root, padding="10 10 10 10", style='TFrame')
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Email input area
        self.input_label = ttk.Label(frame, text="Enter email text:", style='TLabel')
        self.input_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        self.email_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=60, height=10, font=('Arial', 10))
        self.email_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.email_text.configure(background='black', foreground='white')

        # Prediction result display
        self.result_label = ttk.Label(frame, text="Prediction:", style='TLabel')
        self.result_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')

        # Accuracy display
        self.accuracy_label = ttk.Label(frame, text="Accuracy:", style='TLabel')
        self.accuracy_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')

        # Button to toggle detailed results display
        self.toggle_button = ttk.Button(frame, text="Extend Details", style='TButton', command=self.toggle_details)
        self.toggle_button.grid(row=4, column=0, padx=5, pady=5, sticky='w')

        # Detailed results display (initially hidden)
        self.report_label = ttk.Label(frame, text="Classification Report:", style='TLabel')
        self.report_label.configure(background='black', foreground='white')
        self.confusion_label = ttk.Label(frame, text="Confusion Matrix:", style='TLabel')
        self.confusion_label.configure(background='black', foreground='white')

        self.report_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=60, height=10, font=('Courier New', 10))
        self.report_text.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        self.report_text.configure(background='black', foreground='white')

        self.confusion_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=60, height=5,
                                                        font=('Courier New', 10))
        self.confusion_text.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
        self.confusion_text.configure(background='black', foreground='white')

    def toggle_details(self):
        """Toggles visibility of detailed results."""
        if self.report_label.winfo_ismapped() and self.report_text.winfo_ismapped():
            self.report_label.grid_remove()
            self.report_text.grid_remove()
            self.confusion_label.grid_remove()
            self.confusion_text.grid_remove()
            self.toggle_button.config(text="Extend Details")
        else:
            self.report_label.grid(row=5, column=0, padx=5, pady=5, sticky='w')
            self.report_text.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
            self.confusion_label.grid(row=7, column=0, padx=5, pady=5, sticky='w')
            self.confusion_text.grid(row=8, column=0, columnspan=2, padx=5, pady=5)
            self.toggle_button.config(text="Hide Details")
            # Display actual content if not already populated
            if not self.report_text.get(1.0, tk.END).strip():
                self.report_text.insert(tk.END, self.report)
            if not self.confusion_text.get(1.0, tk.END).strip():
                self.confusion_text.insert(tk.END, self.format_confusion_matrix(self.confusion))

    def predict_spam(self, event=None):
        """Performs spam prediction based on current input."""
        email = self.email_text.get("1.0", tk.END).strip()
        if not email:
            messagebox.showwarning("Input Error", "Please enter email text.")
            return

        email = preprocess_text(email)
        new_X = self.vectorizer.transform([email])
        prediction = self.classifier.predict(new_X)[0]

        result = "Spam" if prediction else "Not Spam"
        self.result_label.config(text=f"Prediction: {result}")

    def predict_real_time(self, event):
        """Performs real-time spam prediction as text is typed."""
        email = self.email_text.get("1.0", tk.END).strip()
        if not email:
            self.result_label.config(text="Prediction:")
            self.accuracy_label.config(text="Accuracy:")
            return

        email = preprocess_text(email)
        new_X = self.vectorizer.transform([email])
        prediction = self.classifier.predict(new_X)[0]

        result = "Spam" if prediction else "Not Spam"
        self.result_label.config(text=f"Prediction: {result}")

        # Calculate and update accuracy on test data
        test_data = load_and_preprocess_data()
        X_test = self.vectorizer.transform(test_data['email'])
        y_test = test_data['label'].map({'spam': 1, 'not spam': 0})
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")

    def clear_email_text(self, event=None):
        """Clears the email input area."""
        current_text = self.email_text.get("1.0", tk.END).strip()
        if ' ' in current_text:
            self.email_text.delete("%s-1c wordstart" % tk.INSERT, tk.END)
        else:
            self.email_text.delete("1.0", tk.END)

        self.result_label.config(text="Prediction:")
        self.accuracy_label.config(text="Accuracy:")

    def format_confusion_matrix(self, confusion):
        """Formats the confusion matrix for display."""
        return (f"                  Predicted\n"
                f"              Not Spam     Spam\n"
                f"Actual Not Spam   {confusion[0, 0]:<5}          {confusion[0, 1]:<5}\n"
                f"       Spam       {confusion[1, 0]:<5}          {confusion[1, 1]:<5}")

# Main program execution
if __name__ == "__main__":
    root = tk.Tk()
    app = SpamEmailDetectorApp(root)
    root.mainloop()
