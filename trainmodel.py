from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

# Loop through actions and sequences to load frames
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Load the frame data (allow_pickle=True for potential object arrays)
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            
            # Debugging: Check the shape of each frame loaded
            print(f"Shape of frame {frame_num}: {res.shape}")
            
            # Ensure the frame has a consistent shape (e.g., 63 features per frame)
            if res.shape != (63,):  # Change (63,) to the correct shape of your data
                print(f"Inconsistent frame shape: {res.shape}, skipping this frame.")
                continue  # Skip this frame if it's not the right shape

            # Optional: Pad or truncate frames to make sure they are all the same shape
            if res.shape[0] < 63:
                res = np.pad(res, (0, 63 - res.shape[0]))  # Pad to 63 features if needed

            window.append(res)  # Add the frame to the window
        if len(window) == sequence_length:  # Ensure we have a complete sequence before appending
            sequences.append(window)
            labels.append(label_map[action])

# Convert sequences and labels into arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Create a TensorBoard callback for visualization
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))  # Adjust input shape (30, 63)
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))  # Assuming 'actions' is the label list or array

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Display model summary
model.summary()

# Save the model to a file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
