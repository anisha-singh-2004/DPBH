import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

file_path = "C:\\Users\\anish\\Downloads\\fdt.xlsx"
df = pd.read_excel(file_path)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(df['Data'], df['label'], test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_sequence_length = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_sequence_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_split=0.2)

y_pred = (model.predict(X_test_padded) > 0.5).astype("int32").flatten()
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

model.save('lstm_model.h5')
joblib.dump(tokenizer, 'tokenizer.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

app = Flask(__name__)
CORS(app)

from tensorflow.keras.models import load_model
loaded_model = load_model('lstm_model.h5')
loaded_tokenizer = joblib.load('tokenizer.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    sequence = loaded_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = (loaded_model.predict(padded_sequence) > 0.5).astype("int32").flatten()[0]
    predicted_label = loaded_label_encoder.inverse_transform([prediction])
    return jsonify({'prediction': predicted_label[0]})

if __name__ == '__main__':
    app.run('127.0.0.1', port=5000)
