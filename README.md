## Spam Email Detector

This is a GUI application that detects spam emails using a Naive Bayes classifier. The model is trained on a dataset of labeled email texts. The application allows users to input email text, get real-time predictions, and view detailed model evaluation metrics.

## Features

- **Real-time Spam Detection**: As you type, the application predicts whether the email text is spam or not.
- **Model Evaluation**: Displays the classification report and confusion matrix of the trained model.
- **Keyboard Shortcuts**: Use `Ctrl+Enter` to perform prediction and `Ctrl+Backspace` to clear the email input.

## Installation

1. Clone the repository:
   ```bash
   [https://github.com/Shashank-LS/NLP-Hackathon.git]
   cd NLP-Hackathon
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the root directory of the project:
   - [spam.csv]

## Usage

1. Run the application:
   ```bash
   python spam_detector.py
   ```

2. The GUI will open. Enter email text in the provided input area to get a spam prediction.

## Files

- `spam_detector.py`: The main Python script containing the GUI application code.
- `requirements.txt`: List of Python dependencies required for the project.
- `spam.csv`: The dataset used to train the Naive Bayes classifier.

## Data Preprocessing

- The email texts are preprocessed by converting to lowercase, removing non-alphanumeric characters, and removing extra spaces.

## Model Training

- A Naive Bayes classifier is trained using the `CountVectorizer` on the preprocessed email texts.
- The model is evaluated using accuracy, classification report, and confusion matrix.

## GUI Details

- The GUI is built using `tkinter` and `ttk`.
- The background color is set to black for better readability.
- The `scrolledtext` widget is used for displaying text with scrollbars.

## Real-time Prediction

- As you type in the email text, the application performs real-time prediction and updates the prediction label.
- The accuracy of the model is recalculated and displayed.

## Keyboard Shortcuts

- `Ctrl+Enter`: Perform prediction.
- `Ctrl+Backspace`: Clear the email input area.
