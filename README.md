# Scalable ASL-Recognition with MLP along with LSTM (hybrid model) and Mediapipe
This repository contains a project for recognizing American Sign Language (ASL) gestures using a hybrid model that combines Multi-Layer Perceptron (MLP) and Long Short-Term Memory (LSTM) networks. The project leverages the Mediapipe library for hand landmark detection and extraction of features from video input.

## Requirements (Mandatory)
- Python 3.10.11
- numpy==1.26.4
- tensorflow==2.15.0
- pandas==2.1.4
- protobuf==3.20.3
- opencv-python==4.8.1.78
- mediapipe==0.10.7
- scikit-learn==1.3.2
- pyttsx3==2.90

## Setting Up the Environment
1. Clone the repository to your local machine:
   ```bash
   git clone
    ```
2. Navigate to the project directory:
    ```bash
    cd asl-recog
    ```
3. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    
4. Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run It

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server:**
   ```bash
   python server.py
   ```

3. **Access from your PC:**
   Open your browser and go to: `http://localhost:5000`

4. **Access from Friends' PCs:**
   * Find your local IP address (Run `ipconfig` on Windows, look for IPv4 Address, e.g., `192.168.1.5`).
   * Tell your friends to open `http://192.168.1.5:5000` in their browser (assuming they are on the same Wi-Fi).
   * **Note:** Browsers often block webcam access on non-HTTPS sites unless it is `localhost`. To fix this for a local test:
     * **Chrome:** Go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, enable it, and add `http://192.168.1.X:5000` to the list.
     * **Alternative:** Use a tunneling service like **ngrok** to get a public HTTPS URL:
       ```bash
       ngrok http 5000
       ```
    


## Testing the Requirements
- To ensure all the libraries are correctly installed, run the hand-landmarks.py file to check for the mediapipe feature working.
- Also check the terminal whether the features are being extracted in terms of X,Y and Z along with the total feature count **(21*3=63).**

## Collecting the images for MLP(Multi Layer Perceptron)
- Create a folder named dataset
- Create subfolders for static gestures like A,B,C...
- Run the data-collection.py by adjusting the required character in the code line 6
- Collect around 200-300 images to get more refined training data
- **Ensure the features count is 68**

## Training the MLP Model
- Run the training.py file to train the model on the collected dataset
- The model will be saved as asl_model.h5 after training

## Real-Time ASL Recognition
- Run the asl_recognition.py file to test the real-time ASL recognition using your webcam
- Ensure your webcam is connected and functional
- The model will predict the ASL gestures in real-time and display the results on the video feed
- You can adjust the confidence threshold in the code to improve accuracy


## Collect the images for LSTM(Long Short-Term Memory)
- Create subfolders for dynamic gestures like I,Z,no-gesture in the motion-dataset folder.
- Run the collect-motion-data.py by adjusting the required character using the controls provided in the terminal.
- Collect around 200+ frames of 20 for each character.
- Do it for no-gesture also to increase the model prediction and behaviour
- **Ensure the feature count is 63**

## Training the LSTM Model
- Run the train-lstm.py file to train the model on the collected motion dataset
- The model will be saved as z_j_lstm_model.h5 after training

## Real-Time ASL Recognition for LSTM
- Run the real-time-lstm.py file to test the real-time ASL recognition for dynamic gestures using your webcam
- Ensure your webcam is connected and functional
- The model will predict the ASL dynamic gestures in real-time and display the results on the video feed

## Combining MLP and LSTM for Hybrid Model
- You can combine the predictions from both MLP and LSTM models to create a hybrid ASL recognition system
- This can be done by implementing a decision-making mechanism that considers outputs from both models
- Run the hybrid-realtime.py to check for the hybrid model function.

## Note
- Ensure that your environment has access to a webcam for real-time testing
- Adjust the parameters in the code files as needed for your specific use case
- For better performance, consider using a GPU for training the models

## License
This project is licensed under the MIT License - see the LICENSE file for details.

