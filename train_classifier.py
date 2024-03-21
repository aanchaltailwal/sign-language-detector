import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Check the type of input data
print("Original data type:", type(data_dict['data']))

# Pad or truncate the samples to a fixed length
max_length = 430  # Adjust this according to your data
padded_data = pad_sequences(data_dict['data'], maxlen=max_length, padding='post', truncating='post', dtype='float32')

# Convert each padded sample to a numpy array
data_array = np.array(padded_data)

# Check the shape and type of the converted data array
print("Converted data shape:", data_array.shape)
print("Converted data type:", type(data_array))

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_array)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(scaled_data, data_dict['labels'], test_size=0.2, random_state=42, stratify=data_dict['labels'])

# Define the classifier
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, min_samples_leaf=1)

# Train the model
rf_classifier.fit(x_train, y_train)

# Predictions
y_train_pred = rf_classifier.predict(x_train)
y_test_pred = rf_classifier.predict(x_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the model
with open('best_model.pickle', 'wb') as f:
    pickle.dump(rf_classifier, f)
