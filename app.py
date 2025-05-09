import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'static/uploads'  
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'} 

model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    
    if 'file' not in request.files:
        return redirect(request.url)
    
    f = request.files['file']
    
  
    if f.filename == '':
        return redirect(request.url)
    
  
    if f and allowed_file(f.filename):
        basepath = os.path.dirname(__file__)  
        upload_folder = os.path.join(basepath, 'static', 'uploads') 
        os.makedirs(upload_folder, exist_ok=True)  
        
 
        filename = secure_filename(f.filename)
        file_path = os.path.join(upload_folder, filename)  

       
        f.save(file_path)
        
       
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]

        return render_template('index.html', filename=filename, result=predicted_label)

    return "File type not allowed"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)




