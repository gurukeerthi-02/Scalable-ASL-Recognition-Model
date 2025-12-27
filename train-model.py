import numpy as np
import os

X = [] # our features - get our input data (63 features)
y = [] # our labels - get our output labels (A,B,C...)


# setting our dataset path and fetching the available labels
dataset_path = "dataset"
labels = os.listdir(dataset_path)
print("Labels found: ",labels)


# creating our label map because AI will not understand text (eg: A -> 0)
label_map = {label: idx for idx, label in enumerate(labels)}
print("Label map: ",label_map)



# reading our .npy values and appending them in the X and y array
for label in labels:
    folder_path = os.path.join(dataset_path, label)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)

        X.append(data)
        y.append(label_map[label])

# converting normal py array to numpy array
X = np.array(X)
y = np.array(y)

print("X shape: ",X.shape)
print("y shape: ",y.shape)

# small tip -> ml will use one-hot enocoding
# since we said that the labels will have a numeric value and the np array has 3 class values as of now A,B,C
# the array will be outputed as [1,0,0] -> A, [0,1,0] -> B,etc.

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print("One-hot encoded: ",y)



# NN model is created with the following
# - reLU -> Fast and stable processing
# - Softmax -> Gives probabilities on the output prediction
# - Dense -> Best for numerical data
# - 63 input -> Our feature count is the input



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(68,)),
    Dense(64, activation='relu'),
    Dense(len(labels),activation='softmax')
])



# compiling our model with 
# - Adam -> best general opimizer
# - Categorical_crossentropy -> multi-class classification


model.compile(
    optimizer = 'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']

)


model.fit(
    X,y,
    epochs=30,
    batch_size=32,
    shuffle=True
)


model.save("gesture_model.h5")
print("Model saved successfully !")

