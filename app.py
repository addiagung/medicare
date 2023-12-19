import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pandas as pd

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "static/images/"

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return "." in filename and filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

# Load Keras model for image classification
model = load_model("model.h5", compile=False)

# Read data obat from a CSV file and preprocess the description column
dataobat_df = pd.read_csv("dataobat.csv")
dataobat_df["deskripsi"] = dataobat_df["deskripsi"].str.replace('\n', '<br>')
columns = ["nama", "deskripsi", "manfaat", "efek_samping", "kategori", "link_gambar"]
labels = dataobat_df[columns].values.tolist()

# Define the index route for the API
@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "API fetched successfully",
        },
        "data": None
    }), 200

# Define the route for making detections based on input images
@app.route("/detection", methods=["GET", "POST"])
def detection():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            # Save the input image to the designated folder
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Preprocess the input image for model detection
            img = Image.open(image_path).convert("RGB")
            img = img.resize((150, 150))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Make detections using the loaded model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_names = labels[index]
            confidence_score = prediction[0][index]

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Detection successful",
                },
                "data": {
                    "nama": class_names[0],
                    "deskripsi": class_names[1],
                    "manfaat": class_names[2],
                    "efek_samping": class_names[3],
                    "kategori": class_names[4],
                    "link_gambar": class_names[5],
                    "confidence": float(confidence_score)
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Client-side error"
                },
                "data": None
            }), 400
        
    # Get All Data
    if request.method == "GET":
        return jsonify({
            "status": {
                "code": 200,
                "message": "Successful"
            },
            "data": [{
                "nama": item[0],
                "deskripsi": item[1],
                "manfaat": item[2],
                "efek_samping": item[3],
                "kategori": item[4],
                "link_gambar": item[5],
            } for item in labels]
        }), 200
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405
    
# Define a new route for searching based on keyword
@app.route("/search", methods=["GET"])
def search():
    keyword = request.args.get("keyword")
    if not keyword:
        return jsonify({
            "status": {
                "code": 400,
                "message": "Bad Request - Keyword not provided"
            },
            "data": None
        }), 400

    # Search for 'labels' containing the keyword in 'nama' or 'manfaat'
    keyword_lower = keyword.lower()
    filtered_labels = [item for item in labels if keyword_lower in item[0].lower() or keyword_lower in item[2].lower()]

    if not filtered_labels:
        return jsonify({
            "status": {
                "code": 404,
                "message": f"No results found for keyword '{keyword}'"
            },
            "data": None
    }), 404

    return jsonify({
        "status": {
            "code": 200,
            "message": "Search successful"
        },
        "data": [{
            "nama": item[0],
            "deskripsi": item[1],
            "manfaat": item[2],
            "efek_samping": item[3],
            "kategori": item[4],
            "link_gambar": item[5],
        } for item in filtered_labels]
    }), 200

# Run the Flask application
if __name__ == "__main__":
    app.run()
