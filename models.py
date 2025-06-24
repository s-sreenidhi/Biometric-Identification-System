import os
import cv2
import numpy as np
import pickle
import joblib
import torch
import torch.nn.functional as F
import pywt
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTModel, ViTImageProcessor
from sklearn.metrics.pairwise import cosine_similarity

# === Step 1: Load Saved Models ===
face_model = joblib.load("models/random_forest_face_model.pkl")
with open("models/face_features_and_labels.pkl", 'rb') as f:
    face_data = pickle.load(f)
face_encoder = face_data["label_encoder"]

iris_model = joblib.load("models/rf_iris_model_wtn_vit.pkl")
with open("models/iris_features_wtn_vit.pkl", 'rb') as f:
    iris_data = pickle.load(f)
iris_encoder = iris_data["label_encoder"]

finger_model = joblib.load("models/random_forest_fingerprint_model_cnn_wtn.pkl")
with open("models/fingerprint_features_cnn_wtn.pkl", 'rb') as f:
    finger_data = pickle.load(f)
finger_encoder = finger_data["label_encoder"]

## MEAn vectors
with open("models/face_mean_vectors.pkl", 'rb') as f:
    face_mean_vectors = pickle.load(f)
with open("models/iris_mean_vectors.pkl", 'rb') as f:
    iris_mean_vectors = pickle.load(f)
with open("models/fingerprint_mean_vectors.pkl", 'rb') as f:
    fingerprint_mean_vectors = pickle.load(f)

print("✅ All models and pre-computed data loaded!") # Optional: for confirmation


# === Step 2: Device Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Step 3: Fingerprint CNN Model ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 256)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        return x

finger_cnn_model = SimpleCNN().to(device)
finger_cnn_model.eval()

# === Step 4: ViT Model ===
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device).eval()
vit_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# === Step 5: CNN Model for Face Feature Extraction ===
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn_model = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()

cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Step 6: Feature Extraction Functions ===

def extract_face_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
    img_tensor = cnn_transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_feat = cnn_model(img_tensor).squeeze().cpu().numpy()

    inputs = vit_extractor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        vit_feat = vit_model(**inputs).pooler_output.squeeze().cpu().numpy()

    combined = np.concatenate((cnn_feat, vit_feat))
    return combined

def wavelet_transform(image):
    coeffs = pywt.wavedec2(image, 'haar', level=2)
    flattened = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            for arr in coeff:
                flattened.extend(arr.flatten())
        else:
            flattened.extend(coeff.flatten())
    return np.array(flattened[:4096])

def extract_iris_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_resized = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    wavelet_feat = wavelet_transform(gray)

    inputs = vit_extractor(images=image_resized, return_tensors="pt").to(device)
    with torch.no_grad():
        vit_feat = vit_model(**inputs).pooler_output.squeeze().cpu().numpy()

    combined = np.concatenate((wavelet_feat, vit_feat))
    return combined

transform_finger = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
#
# def extract_fingerprint_features(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         return None
#     image_resized = cv2.resize(image, (224, 224))
#
#     wtn_feat = wavelet_transform(image_resized)[:2048]
#
#     input_tensor = transform_finger(image_resized).unsqueeze(0).to(device)
#     with torch.no_grad():
#         cnn_feat = finger_cnn_model(input_tensor).cpu().numpy().flatten()
#     cnn_feat = cnn_feat[:256]
#
#     combined = np.concatenate((wtn_feat, cnn_feat))
#     return combined

def extract_fingerprint_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image_resized = cv2.resize(image, (224, 224))

    # --- Start of the fix ---
    wtn_feat = wavelet_transform(image_resized)
    # This line ensures the vector is EXACTLY 2048
    wtn_feat_padded = np.pad(wtn_feat, (0, 2048 - len(wtn_feat) % 2048), 'constant')[:2048]

    input_tensor = transform_finger(image_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        cnn_feat = finger_cnn_model(input_tensor).cpu().numpy().flatten()

    # This line ensures the vector is EXACTLY 256
    cnn_feat_padded = np.pad(cnn_feat, (0, 256 - len(cnn_feat)), 'constant')[:256]

    # Combine the padded, safe vectors
    combined = np.concatenate((wtn_feat_padded, cnn_feat_padded))
    # --- End of the fix ---

    return combined
# === Step 7: Main Prediction Function ===
# def predict_user(face_path, iris_path, fingerprint_path):
#     face_features = extract_face_features(face_path)
#     face_pred = face_model.predict([face_features])[0]
#     face_label = face_encoder.inverse_transform([face_pred])[0]
#
#     iris_features = extract_iris_features(iris_path)
#     iris_pred = iris_model.predict([iris_features])[0]
#     iris_label = iris_encoder.inverse_transform([iris_pred])[0]
#
#     fingerprint_features = extract_fingerprint_features(fingerprint_path)
#     finger_pred = finger_model.predict([fingerprint_features])[0]
#     finger_label = finger_encoder.inverse_transform([finger_pred])[0]
#
#     return face_label == iris_label == finger_label
# === Step 7: The NEW, UPGRADED Main Prediction Function ===
def predict_user(face_path, iris_path, fingerprint_path):
    # Extract features from the new, live images
    face_features = extract_face_features(face_path)
    iris_features = extract_iris_features(iris_path)
    fingerprint_features = extract_fingerprint_features(fingerprint_path)

    # Gracefully handle if an image fails to load
    if any(f is None for f in [face_features, iris_features, fingerprint_features]):
        print("Error: Could not read one or more of the uploaded images.")
        return False, "Error", 0

    # Predict who the person is for each modality
    face_pred_idx = face_model.predict([face_features])[0]
    iris_pred_idx = iris_model.predict([iris_features])[0]
    finger_pred_idx = finger_model.predict([fingerprint_features])[0]

    # Decode the prediction to get the label name (e.g., 'person1')
    face_label = face_encoder.inverse_transform([face_pred_idx])[0]
    iris_label = iris_encoder.inverse_transform([iris_pred_idx])[0]
    finger_label = finger_encoder.inverse_transform([finger_pred_idx])[0]

    # --- This is the new logic ---
    # Look up the pre-computed average vector for the predicted person
    face_mean = face_mean_vectors.get(face_label)
    iris_mean = iris_mean_vectors.get(iris_label)
    finger_mean = fingerprint_mean_vectors.get(finger_label)

    # Handle case where a predicted label might not be in our mean vectors
    if any(m is None for m in [face_mean, iris_mean, finger_mean]):
        print("Error: Predicted label not found in pre-computed mean vectors.")
        return False, "Unknown", 0

    # Calculate cosine similarity score (how close is the new image to the average)
    face_score = cosine_similarity([face_features], [face_mean])[0][0]
    iris_score = cosine_similarity([iris_features], [iris_mean])[0][0]
    finger_score = cosine_similarity([fingerprint_features], [finger_mean])[0][0]

    # Average the scores to get the final match percentage
    # Use max(0, ...) to prevent tiny negative scores from floating point errors
    match_percentage = round(max(0, (face_score + iris_score + finger_score) / 3) * 100, 2)
    print("############## 🙂🙂 GOT THIS MATCH % :",match_percentage)
    # Check if all models agreed on the same person for authorization
    authorized = (face_label == iris_label == finger_label)

    # If authorized, the user label is consistent. Otherwise, we can decide what to return.
    # We'll return the face_label as the primary guess.
    final_label = face_label if authorized else "Mismatch"

    if not authorized:
        print(
            f"Identity Mismatch: Face predicts {face_label}, Iris predicts {iris_label}, Fingerprint predicts {finger_label}")

    # Return the three key pieces of information
    return authorized, final_label, match_percentage
