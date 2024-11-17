import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load the Excel file
file_path = "C:\\Users\\anish\\Downloads\\fdt.xlsx"
df = pd.read_excel(file_path)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Data'], df['label'], test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
predictions = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, predictions))

# Save the model using joblib
joblib.dump(model, 'model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# Flask API
app = Flask(__name__)
CORS(app)

# Load the saved model and vectorizer
loaded_model = joblib.load('model.joblib')
loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = tfidf_vectorizer.transform([text])

    # Make prediction
    prediction = loaded_model.predict(processed_text)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run('127.0.0.1', port=5000)
