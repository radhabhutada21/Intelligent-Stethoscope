import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from collections import Counter

# Dataset paths
DATASET_PATH = r"C:\Users\lenovo\.cache\kagglehub\datasets\vbookshelf\respiratory-sound-database\versions\2\Respiratory_Sound_Database\Respiratory_Sound_Database"
AUDIO_PATH = os.path.join(DATASET_PATH, "audio_and_txt_files")
CSV_PATH = os.path.join(DATASET_PATH, "patient_diagnosis.csv")

# Extract MFCC features from audio file
def extract_features(file_path, max_pad_len=862):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset and filter rare classes
def load_dataset():
    diagnosis_df = pd.read_csv(CSV_PATH, header=None, names=['patient_id', 'disease'])
    print("Total Patients in CSV:", len(diagnosis_df))

    features, labels = [], []

    for file in os.listdir(AUDIO_PATH):
        if file.endswith(".wav"):
            patient_id = int(file.split('_')[0])
            disease_row = diagnosis_df[diagnosis_df['patient_id'] == patient_id]
            if not disease_row.empty:
                disease = disease_row['disease'].values[0]
                file_path = os.path.join(AUDIO_PATH, file)
                mfccs = extract_features(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(disease)

    print("Total audio files processed:", len(features))

    # Remove rare classes
    counts = Counter(labels)
    valid_classes = [cls for cls, count in counts.items() if count >= 2]
    features, labels = zip(*[(f, l) for f, l in zip(features, labels) if l in valid_classes])

    print("\nClass distribution after filtering:")
    print(Counter(labels))

    return np.array(features), np.array(labels)

# Build CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main script
if __name__ == "__main__":
    X, y = load_dataset()

    X = X[..., np.newaxis]
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X_train.shape[1:], y_encoded.shape[1])

    history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {acc*100:.2f}%")

    model.save("model.h5")
    print("\nModel saved as model.h5")
