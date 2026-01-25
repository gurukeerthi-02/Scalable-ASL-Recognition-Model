# Scalable ASL Recognition System (Backend & Models)

This repository hosts the backend and machine learning components for a scalable American Sign Language (ASL) recognition system. The system employs a hybrid architecture combining **Multi-Layer Perceptron (MLP)** for static gestures and **Long Short-Term Memory (LSTM)** networks for dynamic gestures, leveraging **Mediapipe** for robust hand landmark extraction.

## 📂 Project Structure

- **`server.py`**: The main Flask + Socket.IO backend server. Handles real-time video stream processing and inference.
- **`utils/improved/`**: Contains the latest, robust scripts for the development workflow (collection, training, testing).
- **`utils/models/`**: Storage for trained model artifacts.(empty)
- **`utils/dataset_merged/`**: The consolidated dataset used for training.(empty)

---

## 🧠 Trained Models

### 1. Static Gesture Model (MLP)
- **Architecture**: Deep Neural Network (MLP) with Dropout and BatchNormalization.
- **Input**: 68 normalized features (Relative (x,y,z) coordinates + Euclidean distances between fingertips and palm).
- **Training Strategy**: **Person-Based Split**.
    - Ensures that the training, validation, and test sets contain data from *different* people to prevent overfitting and ensure real-world generalization.
- **Classes**: A-Z (Static alphabet excluding J and Z).

### 2. Dynamic Gesture Model (LSTM)
- **Architecture**: LSTM for temporal sequence analysis.
- **Input**: Sequence of 30 frames (x, y, z coordinates).
- **Classes**: Dynamic words like "Hello", "J", "Z".

---

## 🚀 Running the Backend

### 1. Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Model Setup
Ensure the trained models are placed where `server.py` expects them (create a `models` folder in the root if it doesn't exist):
```bash
# Example setup
mkdir models
cp utils/models/static_model_person_split_v1.h5 models/gesture_model.h5
# Copy the LSTM model similarly
```

### 3. Start the Server
Run the Flask server to start the backend with Socket.IO support:
```bash
python server.py
```
The server will start on `http://0.0.0.0:5000`.

---

## 🛠️ Development Workflow

The `utils/improved/` directory contains the complete pipeline for building and improving the models.

### Step 1: Data Collection
Collect high-quality data using the portable collector. This script captures landmarks for specific gestures.
```bash
# Coordinate to utils/improved directory
cd utils/improved

# Run the collector
python collect-data-portable.py
```
*Follow the on-screen prompts to select the label and number of samples.*

### Step 2: Dataset Merging
Consolidate data from multiple sessions or different people into a single master dataset.
```bash
python merge-datasets.py
```
*Output*: Merged data will be stored in `../dataset_merged`.

### Step 3: Training (Person-Based)
Train the MLP model using the "Person-Based Split" strategy. This script automatically handles augmentation and splits data by person ID.
```bash
python person-based-training-static-model.py
```
*Output*: A new model file saved to `../models/static_model_person_split_v1.h5`.

### Step 4: Comprehensive Testing
Validate the model against a robust testing suite to measure accuracy, confusion matrices, and per-class performance.
```bash
python comprehensive_testing_static.py
```

---

## ⚙️ Technical Details

### Feature Extraction
- **Static**: 21 landmarks × 3 (x, y, z) + 4 fingertip distances + 1 thumb-index distance = **68 Features**.
- **Dynamic**: 21 landmarks × 3 (x, y, z) = **63 Features** per frame.

### Overfitting Prevention
We strictly use **Person-Based Splitting** instead of random splitting. This guarantees that the model learns generalized ASL features rather than memorizing the specific hand shapes or camera angles of a single user.
