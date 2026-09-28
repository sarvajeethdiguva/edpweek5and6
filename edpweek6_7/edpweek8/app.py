from flask import Flask, request, render_template_string
from PIL import Image
import tensorflow as tf
import numpy as np
from pathlib import Path
import base64
import io


app = Flask(__name__)


# ============================================================
# LOAD TRAINED CNN MODEL
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "plant_disease_cnn.keras"
)

model = None
model_error = None

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    print("Model loaded successfully")
    print("Model path:", MODEL_PATH)

except Exception as e:
    model_error = str(e)
    print("WARNING: Model could not be loaded")
    print(model_error)


# ============================================================
# PLANT DISEASE CLASS NAMES
# ============================================================

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


# ============================================================
# HTML
# ============================================================

HTML = """
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Plant Disease Detector</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #f4f7f4;
            color: #222;
        }

        .container {
            width: 90%;
            max-width: 900px;
            margin: 40px auto;
        }

        .card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.10);
        }

        h1 {
            text-align: center;
            color: #207a3c;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }

        .upload-box {
            border: 2px dashed #4caf50;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            background: #f9fff9;
        }

        input[type="file"] {
            margin: 20px 0;
            width: 100%;
        }

        button {
            background: #2e8b57;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }

        button:hover {
            background: #256f46;
        }

        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 12px;
            background: #eef8ef;
        }

        .result h2 {
            color: #207a3c;
        }

        .prediction {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }

        .confidence {
            font-size: 18px;
            margin-bottom: 20px;
        }

        .uploaded-image {
            max-width: 100%;
            max-height: 400px;
            border-radius: 12px;
            margin-top: 15px;
        }

        .top3 {
            margin-top: 20px;
        }

        .top3 li {
            margin: 8px 0;
        }

        .warning {
            margin-top: 20px;
            padding: 15px;
            background: #fff3cd;
            color: #856404;
            border-radius: 8px;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            background: #f8d7da;
            color: #721c24;
            border-radius: 8px;
        }

        .footer {
            text-align: center;
            margin-top: 25px;
            color: #777;
            font-size: 14px;
        }

    </style>

</head>


<body>

<div class="container">

    <div class="card">

        <h1>🌱 Plant Disease Detector</h1>

        <p class="subtitle">
            Upload a plant leaf image to detect possible diseases
            using a CNN model.
        </p>


        <div class="upload-box">

            <form
                method="POST"
                enctype="multipart/form-data"
            >

                <label>
                    <strong>Select a plant leaf image</strong>
                </label>

                <br>

                <input
                    type="file"
                    name="image"
                    accept="image/*"
                    required
                >

                <br>

                <button type="submit">
                    🔍 Detect Disease
                </button>

            </form>

        </div>


        {% if prediction %}

        <div class="result">

            <h2>Prediction Result</h2>

            <div class="prediction">
                {{ prediction }}
            </div>

            <div class="confidence">
                Confidence: <strong>{{ confidence }}%</strong>
            </div>


            {% if image_data %}

            <h3>Uploaded Image</h3>

            <img
                class="uploaded-image"
                src="data:image/jpeg;base64,{{ image_data }}"
                alt="Uploaded plant image"
            >

            {% endif %}


            {% if top3 %}

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

            {% endif %}


            {% if confidence < 60 %}

            <div class="warning">

                ⚠️ Low confidence prediction.
                Please upload a clearer leaf image or consult
                an agricultural expert before taking action.

            </div>

            {% endif %}

        </div>

        {% endif %}


        {% if error %}

        <div class="error">

            <strong>Error:</strong>
            {{ error }}

        </div>

        {% endif %}


        {% if model_error %}

        <div class="warning">

            The web application is running, but the trained model
            could not be loaded.

            <br><br>

            Model loading information is available in the
            deployment logs.

        </div>

        {% endif %}


        <div class="footer">

            Plant Disease Detection using Convolutional Neural Network

        </div>

    </div>

</div>

</body>

</html>
"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    top3 = []
    image_data = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please select an image."

            return render_template_string(
                HTML,
                prediction=prediction,
                confidence=confidence,
                top3=top3,
                image_data=image_data,
                error=error,
                model_error=model_error
            )


        try:

            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            image = Image.open(file).convert("RGB")


            # ------------------------------------------------
            # PREPARE IMAGE FOR CNN
            # ------------------------------------------------

            image_resized = image.resize((224, 224))

            image_array = np.array(
                image_resized,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )


            # ------------------------------------------------
            # MODEL CHECK
            # ------------------------------------------------

            if model is None:

                prediction = "Model unavailable"
                confidence = 0
                top3 = []

            else:

                # --------------------------------------------
                # MAKE PREDICTION
                # --------------------------------------------

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]


                # --------------------------------------------
                # SAFETY CHECK
                # --------------------------------------------

                if len(predictions) != len(CLASS_NAMES):

                    raise ValueError(
                        "Model output contains "
                        + str(len(predictions))
                        + " classes, but CLASS_NAMES contains "
                        + str(len(CLASS_NAMES))
                        + " classes."
                    )


                # --------------------------------------------
                # TOP 3 PREDICTIONS
                # --------------------------------------------

                top_indices = np.argsort(
                    predictions
                )[-3:][::-1]


                top3 = [

                    (
                        CLASS_NAMES[i],
                        round(
                            float(predictions[i]) * 100,
                            2
                        )
                    )

                    for i in top_indices

                ]


                # --------------------------------------------
                # BEST PREDICTION
                # --------------------------------------------

                best_index = top_indices[0]

                prediction = CLASS_NAMES[best_index]

                confidence = round(
                    float(predictions[best_index]) * 100,
                    2
                )


            # ------------------------------------------------
            # CONVERT IMAGE TO BASE64
            # ------------------------------------------------

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="JPEG"
            )

            image_data = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")


        except Exception as e:

            error = str(e)

            print("Prediction error:")
            print(error)


    return render_template_string(

        HTML,

        prediction=prediction,

        confidence=confidence,

        top3=top3,

        image_data=image_data,

        error=error,

        model_error=model_error

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )