from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'detection//uploads'

# Load the YOLO model once at the start
model = YOLO('detection//script//best.pt')

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Save the file to the uploads directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Run YOLO model on the uploaded image
        results = model.predict(source=file_path)
        
        # Get the annotated image as a NumPy array
        annotated_img = results[0].plot()
        
        # Convert the NumPy array to a PIL image and send it to the client
        img = Image.fromarray(annotated_img.astype(np.uint8))
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
