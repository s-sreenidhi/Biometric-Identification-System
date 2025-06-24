import os
import pickle
import numpy as np
print(os.getcwd())
# List of your biometric types
modalities = ['face', 'iris', 'fingerprint']

print("🚀 Starting pre-computation of mean vectors...")

for modality in modalities:
    # Adjust filenames to match your saved files
    if modality == 'face':
        features_path = "../models/face_features_and_labels.pkl"
    elif modality == 'iris':
        features_path = "../models/iris_features_wtn_vit.pkl"
    else:  # fingerprint
        features_path = "../models/fingerprint_features_cnn_wtn.pkl"

    print(f"Processing {modality}...")

    # Load the feature data you saved during training
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    # These are the correct keys from your saved files!
    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])  # Use the ENCODED labels
    encoder = data['label_encoder']

    mean_vectors = {}
    # Loop through each person/class ('person1', 'person2', etc.)
    for i, class_name in enumerate(encoder.classes_):
        # Find all training vectors that belong to this person
        class_vectors = X_train[y_train == i]

        # Calculate the mean (the "average face/iris/fingerprint")
        if len(class_vectors) > 0:
            mean_vectors[class_name] = np.mean(class_vectors, axis=0)

    # Save the computed means to a new, clean file
    output_path = f"../models/{modality}_mean_vectors.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(mean_vectors, f)

    print(f"✅ Saved mean vectors for {modality} to {output_path}")

print("\n✨ All mean vectors computed and saved successfully! You're ready to predict.")