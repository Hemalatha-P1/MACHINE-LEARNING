import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

# Suppress TensorFlow logging except for errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = warnings, 2 = errors, 3 = only critical

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the model
with open('model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

# Labels for classification
labels = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-']

def classify_image(img_path):
    test_image = image.load_img(img_path, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = loaded_model.predict(test_image)
    prediction_index = np.argmax(result[0])
    prediction = labels[prediction_index]
    return prediction

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':  # Example credentials
            session['logged_in'] = True
            return redirect(url_for('classify'))
        else:
            flash('Invalid Credentials. Please try again.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            prediction = classify_image(file_path)

            return render_template('result.html', prediction=prediction, img_path=file.filename)

    return render_template('classify.html')

if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')

    # Run Flask application
    app.run(debug=False, port=8080)  # Disable debug mode if needed
