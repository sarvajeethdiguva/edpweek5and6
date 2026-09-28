from flask import Flask, request, render_template_string
from PIL import Image
import tensorflow as tf
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load trained CNN model
# Load trained CNN model
MODEL_PATH = Path(__file__).resolve().parents[2] / "plant_disease_cnn.keras"

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )
    print("Model loaded successfully")
except Exception as e:
    print("WARNING: Model could not be loaded:", e)
    model = None

# Plant disease class names
CLASS_NAMES = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Detector</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f2f7f2;
            color: #222;
        }

        .container {
            width: 92%;
            max-width: 600px;
            margin: 30px auto;
            background: white;
            padding: 25px;
            border-radius: 18px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }

        h1 {
            margin-bottom: 8px;
            font-size: 30px;
        }

        .subtitle {
            color: #666;
            margin-bottom: 25px;
        }

        input[type="file"] {
            width: 100%;
            padding: 15px;
            margin: 15px 0;
            border: 2px dashed #aaa;
            border-radius: 12px;
            background: #fafafa;
        }

        button {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 10px;
            background: #2e7d32;
            color: white;
            font-size: 18px;
            cursor: pointer;
        }

        button:hover {
            background: #1b5e20;
        }

        img {
            max-width: 100%;
            max-height: 350px;
            margin-top: 20px;
            border-radius: 12px;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            background: #eef7ee;
        }

        .result h2 {
            margin-top: 0;
        }

        .confidence {
            font-size: 20px;
            font-weight: bold;
        }

        .top3 {
            text-align: left;
            margin-top: 20px;
        }

        .top3 li {
            margin: 10px 0;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>🌿 Plant Disease Detector</h1>

    <p class="subtitle">
        Upload a plant leaf image to detect its disease.
    </p>

    <form method="POST" enctype="multipart/form-data">

        <input
            type="file"
            name="image"
            accept="image/*"
            required
        >

        <button type="submit">
            🔍 Detect Disease
        </button>

    </form>

    {% if image_data %}
        <img src="data:image/jpeg;base64,{{ image_data }}">
    {% endif %}

    {% if prediction %}
        <div class="result">

            <h2>Prediction</h2>

            <p>
                <strong>{{ prediction }}</strong>
            </p>

            <p class="confidence">
                Confidence: {{ confidence }}%
            </p>

            <div class="top3">
                <h3>Top 3 Predictions</h3>

                <ol>
                    {% for name, score in top3 %}
                        <li>
                            <strong>{{ name }}</strong>
                            — {{ score }}%
                        </li>
                    {% endfor %}
                </ol>
            </div>

        </div>
    {% endif %}

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    top3 = None
    image_data = None

    if request.method == "POST":

        file = request.files.get("image")

        if file:

            # Read uploaded image
            image = Image.open(file).convert("RGB")

            # Prepare image for CNN
            image_resized = image.resize((224, 224))
            image_array = np.array(image_resized)
            image_array = np.expand_dims(image_array, axis=0)

           # Make prediction
if model is None:
    prediction = "Model unavailable"
    confidence = 0
    top3 = []
else:
    predictions = model.predict(image_array, verbose=0)[0]

    # Get top 3 predictions
    top_indices = np.argsort(predictions)[-3:][::-1]

    top3 = [
        (
            CLASS_NAMES[i],
            round(float(predictions[i]) * 100, 2)
        )
        for i in top_indices
    ]

    # Best prediction
    best_index = top_indices[0]

    prediction = CLASS_NAMES[best_index]
    confidence = round(float(predictions[best_index]) * 100, 2)

            # Display uploaded image
            import base64
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_data = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        top3=top3,
        image_data=image_data
    )


if __name__ == "__main__":
    app.run(debug=True)