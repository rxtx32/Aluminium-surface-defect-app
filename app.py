# app.py
from flask import Flask, request, render_template_string, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = r"D:\Ritu\Projects\Surface_Defect_App\model\aluminum_defect_model_final.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)
model.make_predict_function()

print("Model loaded successfully!")

# -----------------------------
# Class Names
# -----------------------------
class_names = [
    'Be injured by a collision', 'Clean sample', 'Coating cracking',
    'Convex powder', 'Dirty spot', 'Drain bottom', 'Orange peel',
    'The transverse strip is dented', 'non-conducting', 'pitting', 'scuffing'
]

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_defect(img_path):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100
    return predicted_class, confidence

# -----------------------------
# HTML Template
# -----------------------------
HTML_TEMPLATE = '''
<!doctype html>
<html>
<head>
    <title>Aluminium Surface Defect Detection</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            text-align: center;
            color: dark-blue;

            background-image: url("{{ url_for('static', filename='background.jpg') }}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        .main-title {
            margin-top: 50px;
            font-size: 34px;
            font-weight: bold;
            text-shadow: 2px 2px 6px dark-blue;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            color: #333;
            padding: 40px;
            border-radius: 15px;
            width: 420px;
            margin: 40px auto;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }

        .card h2 {
            margin-bottom: 20px;
        }

        input[type="file"] {
            margin: 15px 0;
        }

        input[type="submit"] {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            background-color: #8b5e3c;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }

        input[type="submit"]:hover {
            background-color: #6f4a2f;
        }

        img {
            max-width: 250px;
            margin-top: 15px;
            border-radius: 10px;
        }

        .result {
            margin-top: 20px;
            background: #f9f3ea;
            padding: 15px;
            border-radius: 10px;
        }

        .confidence {
            color: green;
            font-weight: bold;
        }
    </style>
</head>
<body>

    <div class="main-title">
        Aluminium Surface Defect Detection
    </div>

    <div class="card">
        <h2>Upload Image for Prediction</h2>

        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br>
            <input type="submit" value="Upload & Predict">
        </form>

        {% if filename %}
        <div class="result">
            <h3>Uploaded Image:</h3>
            <img src="{{ url_for('static', filename='uploads/' + filename) }}">
            <p><strong>Predicted Defect:</strong> {{ prediction }}</p>
            <p><strong>Confidence:</strong>
                <span class="confidence">{{ confidence }}%</span>
            </p>
        </div>
        {% endif %}
    </div>

</body>
</html>
'''

# -----------------------------
# Route
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            prediction, confidence = predict_defect(file_path)

            return render_template_string(
                HTML_TEMPLATE,
                filename=filename,
                prediction=prediction,
                confidence=f"{confidence:.2f}"
            )

    return render_template_string(HTML_TEMPLATE, filename=None)

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)