
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Checking Static Model...")
static_path = os.path.join(BASE_DIR, "utils/models/static_model_person_split_v7.h5")
if os.path.exists(static_path):
    model = load_model(static_path)
    print(f"Static Model Input Shape: {model.input_shape}")
else:
    print("Static model not found")

print("\nChecking Dynamic Model...")
dynamic_path = os.path.join(BASE_DIR, "utils/models/dynamic_model_final.h5")
if os.path.exists(dynamic_path):
    model = load_model(dynamic_path)
    print(f"Dynamic Model Input Shape: {model.input_shape}")
else:
    print("Dynamic model not found")
