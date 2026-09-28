from flask import Flask, request, render_template_string
from PIL import Image
import tensorflow as tf
import numpy as np
from pathlib import Path
import base64
import io
import os


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


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
# FIND TRAINED CNN MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILENAME = "plant_disease_cnn.keras"

model = None
model_error = None
MODEL_PATH = None


def find_model():
    """
    Search for plant_disease_cnn.keras in the project
    and nearby folders.
    """

    search_locations = [
        BASE_DIR,
        BASE_DIR / "models",
        BASE_DIR / "model",
        BASE_DIR / "saved_model",
        BASE_DIR / "saved_models",
        BASE_DIR.parent,
        BASE_DIR.parent / "models",
        BASE_DIR.parent / "model",
        BASE_DIR.parent.parent,
        BASE_DIR.parent.parent / "models",
        BASE_DIR.parent.parent / "model",
    ]

    # First check the known locations
    for location in search_locations:
        model_file = location / MODEL_FILENAME

        if model_file.exists() and model_file.is_file():
            return model_file

    # If not found, recursively search the project
    try:
        for model_file in BASE_DIR.rglob(MODEL_FILENAME):
            if model_file.is_file():
                return model_file
    except Exception:
        pass

    return None


# ============================================================
# LOAD MODEL
# ============================================================

def find_model():
    """
    Find the trained plant disease model.
    The application uses the H5 version of the model.
    """

    possible_paths = [
        Path(__file__).resolve().parent / "plant_disease_cnn.h5",
        Path(__file__).resolve().parent / "edpweek6_7" / "edpweek8" / "plant_disease_cnn.h5",
        Path.cwd() / "plant_disease_cnn.h5",
    ]

    # Check known locations first
    for path in possible_paths:
        if path.exists():
            return str(path)

    # Search the project directory recursively
    project_root = Path(__file__).resolve().parents[2]

    for path in project_root.rglob("plant_disease_cnn.h5"):
        if path.is_file():
            return str(path)

    return None


# ============================================================
# HTML PAGE
# ============================================================

HTML = """
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>Plant Disease Detector</title>


    <style>

        * {
            box-sizing: border-box;
        }


        body {

            margin: 0;

            padding: 0;

            font-family:
                Arial,
                Helvetica,
                sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #e8f5e9,
                    #f1f8e9
                );

            color: #222;

        }


        .container {

            width: 90%;

            max-width: 1000px;

            margin: 40px auto;

        }


        .header {

            text-align: center;

            margin-bottom: 30px;

        }


        .header h1 {

            color: #19733b;

            font-size: 42px;

            margin-bottom: 10px;

        }


        .header p {

            font-size: 18px;

            color: #555;

        }


        .card {

            background: white;

            border-radius: 18px;

            padding: 30px;

            margin-bottom: 25px;

            box-shadow:
                0 8px 25px
                rgba(0, 0, 0, 0.08);

        }


        .upload-area {

            border: 3px dashed #43a047;

            border-radius: 15px;

            padding: 45px 20px;

            text-align: center;

            background: #f8fff9;

        }


        .upload-area h2 {

            color: #2e7d32;

            margin-bottom: 10px;

        }


        .upload-area p {

            color: #666;

            margin-bottom: 25px;

        }


        input[type="file"] {

            margin: 15px 0;

            padding: 10px;

            width: 100%;

            max-width: 500px;

        }


        button {

            background: #2e7d32;

            color: white;

            border: none;

            padding: 14px 35px;

            border-radius: 8px;

            font-size: 17px;

            cursor: pointer;

            margin-top: 10px;

        }


        button:hover {

            background: #1b5e20;

        }


        .result {

            background: #eef9f0;

            border-radius: 15px;

            padding: 30px;

        }


        .result h2 {

            color: #19733b;

            margin-top: 0;

        }


        .prediction {

            font-size: 28px;

            font-weight: bold;

            color: #1b5e20;

            margin: 15px 0;

        }


        .confidence {

            font-size: 20px;

            margin-bottom: 25px;

        }


        .image-container {

            margin-top: 25px;

        }


        .uploaded-image {

            width: 100%;

            max-width: 450px;

            max-height: 450px;

            object-fit: contain;

            border-radius: 12px;

            border: 2px solid #ddd;

        }


        .top3 {

            margin-top: 30px;

        }


        .top3 h3 {

            color: #333;

        }


        .top3 ol {

            padding-left: 25px;

        }


        .top3 li {

            margin: 12px 0;

            font-size: 17px;

        }


        .warning {

            background: #fff4cc;

            border-left: 5px solid #f0ad00;

            padding: 15px;

            margin-top: 25px;

            border-radius: 8px;

            color: #795548;

        }


        .error {

            background: #ffebee;

            border-left: 5px solid #e53935;

            padding: 15px;

            margin-top: 20px;

            border-radius: 8px;

            color: #b71c1c;

        }


        .model-status {

            text-align: center;

            font-size: 14px;

            color: #777;

            margin-top: 20px;

        }


        .footer {

            text-align: center;

            margin-top: 30px;

            color: #777;

            font-size: 14px;

        }


    </style>

</head>


<body>


<div class="container">


    <div class="header">

        <h1>🌿 Plant Disease Detector</h1>

        <p>
            Upload a plant leaf image to detect
            possible diseases using a CNN model.
        </p>

    </div>


    <div class="card">


        <div class="upload-area">

            <h2>Upload Leaf Image</h2>

            <p>
                Select a clear image of a plant leaf
            </p>


            <form method="POST"
                  enctype="multipart/form-data">

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


    </div>


    {% if prediction is not none %}

    <div class="card">


        <div class="result">


            <h2>Prediction Result</h2>


            {% if prediction == "Model unavailable" %}

                <div class="prediction">
                    Model unavailable
                </div>

                <div class="confidence">
                    Confidence: 0%
                </div>


                {% if model_error %}

                <div class="error">

                    <strong>Model Error:</strong><br>

                    {{ model_error }}

                </div>

                {% endif %}


            {% else %}

                <div class="prediction">

                    {{ prediction }}

                </div>


                <div class="confidence">

                    Confidence:
                    <strong>{{ confidence }}%</strong>

                </div>


                {% if confidence < 60 %}

                <div class="warning">

                    ⚠️ Low confidence prediction.
                    Please upload a clearer leaf image
                    or consult an agricultural expert
                    before taking action.

                </div>

                {% elif confidence < 80 %}

                <div class="warning">

                    ⚠️ Moderate confidence prediction.
                    Consider uploading a clearer image
                    for better accuracy.

                </div>

                {% endif %}


                {% if top3 %}

                <div class="top3">

                    <h3>Top 3 Predictions</h3>

                    <ol>

                        {% for item in top3 %}

                        <li>

                            <strong>
                                {{ item[0] }}
                            </strong>

                            -
                            {{ item[1] }}%

                        </li>

                        {% endfor %}

                    </ol>

                </div>

                {% endif %}

            {% endif %}


            {% if image_data %}

            <div class="image-container">

                <h3>Uploaded Image</h3>

                <img
                    src="data:image/jpeg;base64,{{ image_data }}"
                    class="uploaded-image"
                    alt="Uploaded plant leaf"
                >

            </div>

            {% endif %}


            {% if error %}

            <div class="error">

                <strong>Prediction Error:</strong><br>

                {{ error }}

            </div>

            {% endif %}


        </div>


    </div>

    {% endif %}


    <div class="model-status">

        {% if model_loaded %}

            🟢 CNN model loaded successfully

        {% else %}

            🔴 CNN model is not loaded

        {% endif %}

    </div>


    <div class="footer">

        Plant Disease Detection System

    </div>


</div>


</body>

</html>
"""


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image):

    """
    Prepare uploaded image for CNN prediction.
    """

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Most CNN models trained using image generators
    # expect pixel values between 0 and 1.
    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# GET MODEL PREDICTIONS
# ============================================================

def get_predictions(image_array):

    """
    Run the CNN model and return probabilities.
    """

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predictions = np.asarray(
        predictions
    ).squeeze()

    # Make sure predictions are one-dimensional
    predictions = predictions.reshape(-1)

    # Check model output
    if len(predictions) != len(CLASS_NAMES):

        raise ValueError(
            "Model output contains "
            + str(len(predictions))
            + " classes, but CLASS_NAMES contains "
            + str(len(CLASS_NAMES))
            + " classes."
        )

    # If model output does not look like probabilities,
    # convert logits to probabilities using softmax.
    if (
        np.any(predictions < 0)
        or
        np.any(predictions > 1)
        or
        not np.isclose(
            np.sum(predictions),
            1.0,
            atol=0.05
        )
    ):

        exp_values = np.exp(
            predictions - np.max(predictions)
        )

        predictions = (
            exp_values
            /
            np.sum(exp_values)
        )

    return predictions


# ============================================================
# HOME ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    prediction = None

    confidence = 0

    top3 = []

    image_data = None

    error = None


    if request.method == "POST":

        file = request.files.get("image")


        # ----------------------------------------------------
        # CHECK FILE
        # ----------------------------------------------------

        if file is None or file.filename == "":

            error = "Please select an image."

            return render_template_string(
                HTML,
                prediction=prediction,
                confidence=confidence,
                top3=top3,
                image_data=image_data,
                error=error,
                model_error=model_error,
                model_loaded=(model is not None)
            )


        try:

            # ------------------------------------------------
            # OPEN IMAGE
            # ------------------------------------------------

            image = Image.open(file).convert("RGB")


            # ------------------------------------------------
            # SAVE IMAGE FOR DISPLAY
            # ------------------------------------------------

            display_buffer = io.BytesIO()

            image.save(
                display_buffer,
                format="JPEG"
            )

            image_data = base64.b64encode(
                display_buffer.getvalue()
            ).decode("utf-8")


            # ------------------------------------------------
            # CHECK MODEL
            # ------------------------------------------------

            if model is None:

                prediction = "Model unavailable"

                confidence = 0

                top3 = []


            else:

                # --------------------------------------------
                # PREPARE IMAGE
                # --------------------------------------------

                image_array = prepare_image(
                    image
                )


                # --------------------------------------------
                # MAKE PREDICTION
                # --------------------------------------------

                predictions = get_predictions(
                    image_array
                )


                # --------------------------------------------
                # TOP 3 PREDICTIONS
                # --------------------------------------------

                top_indices = np.argsort(
                    predictions
                )[-3:][::-1]


                top3 = []

                for index in top_indices:

                    class_name = CLASS_NAMES[
                        int(index)
                    ]

                    class_confidence = round(
                        float(
                            predictions[index]
                        ) * 100,
                        2
                    )

                    top3.append(
                        (
                            class_name,
                            class_confidence
                        )
                    )


                # --------------------------------------------
                # BEST PREDICTION
                # --------------------------------------------

                best_index = int(
                    np.argmax(predictions)
                )


                prediction = CLASS_NAMES[
                    best_index
                ]


                confidence = round(
                    float(
                        predictions[best_index]
                    ) * 100,
                    2
                )


        except Exception as e:

            error = str(e)

            print("==============================================")
            print("PREDICTION ERROR")
            print(error)
            print("==============================================")


    return render_template_string(

        HTML,

        prediction=prediction,

        confidence=confidence,

        top3=top3,

        image_data=image_data,

        error=error,

        model_error=model_error,

        model_loaded=(model is not None)

    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    if model is not None:

        return {
            "status": "ok",
            "model": "loaded",
            "model_path": str(MODEL_PATH)
        }

    return {
        "status": "warning",
        "model": "not loaded",
        "error": model_error
    }


# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )