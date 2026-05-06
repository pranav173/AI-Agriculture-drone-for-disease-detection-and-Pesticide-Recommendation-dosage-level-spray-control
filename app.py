import cv2
import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, db, storage

# =========================
# 🔹 1. FIREBASE SETUP
# =========================

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://drone-1fea5-default-rtdb.asia-southeast1.firebasedatabase.app',
        'storageBucket': 'drone-1fea5.appspot.com'
    })

print("Firebase Connected")

# =========================
# 🔹 2. LOAD MODEL
# =========================

model = tf.keras.models.load_model("model.h5")
print(" Model Loaded")

# =========================
# 🔹 3. DISEASE DATABASE
# =========================

disease_info = {
    "Leaf Blight": {
        "pesticide": "Mancozeb",
        "dosage": 2
    },
    "Powdery Mildew": {
        "pesticide": "Sulfur",
        "dosage": 1.5
    },
    "Healthy": {
        "pesticide": "None",
        "dosage": 0
    }
}

classes = ["Leaf Blight", "Powdery Mildew", "Healthy"]

# =========================
# 🔹 4. PREDICTION FUNCTION
# =========================

def predict_and_recommend(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(" Image not found. Check path!")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_idx = np.argmax(pred)

    disease = classes[class_idx]
    pesticide = disease_info[disease]["pesticide"]
    dosage = disease_info[disease]["dosage"]

    return {
        "disease": disease,
        "pesticide": pesticide,
        "dosage_ml_per_litre": dosage
    }

# =========================
# 🔹 5. SPRAY CALCULATION
# =========================

def compute_spray_plan(dosage_ml_per_litre, tank_litres, flow_ml_per_sec):
    total_pesticide = dosage_ml_per_litre * tank_litres
    spray_time = (tank_litres * 1000) / flow_ml_per_sec

    return {
        "total_pesticide_ml": total_pesticide,
        "spray_time_sec": spray_time
    }

# =========================
# 🔹 6. UPLOAD TO FIREBASE
# =========================

def upload_to_firebase(image_path, result, spray):
    bucket = storage.bucket()

    blob = bucket.blob("images/" + image_path)
    blob.upload_from_filename(image_path)
    blob.make_public()

    image_url = blob.public_url

    ref = db.reference("drone_data")

    ref.set({
        "image_url": image_url,
        "disease": result["disease"],
        "pesticide": result["pesticide"],
        "dosage_ml_per_litre": result["dosage_ml_per_litre"],
        "total_pesticide_ml": spray["total_pesticide_ml"],
        "spray_time_sec": spray["spray_time_sec"]
    })

    print(" Uploaded to Firebase")

# =========================
# 🔹 7. MAIN EXECUTION
# =========================

if __name__ == "__main__":

    image_path = "test.jpg"   # change this to your image

    # Step 1: Predict
    result = predict_and_recommend(image_path)
    print("\n Prediction Result:")
    print(result)

    # Step 2: Spray Plan
    spray = compute_spray_plan(
        result["dosage_ml_per_litre"],
        tank_litres=5,
        flow_ml_per_sec=20
    )

    print("\n🚿 Spray Plan:")
    print(spray)

    # Step 3: Upload
    upload_to_firebase(image_path, result, spray)

    print("\n Process Completed Successfully")