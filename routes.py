from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models import predict_user
import os
from werkzeug.utils import secure_filename
import random  # New import for match percentage simulation

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'face' not in request.files or 'iris' not in request.files or 'finger' not in request.files:
            flash('All three files must be uploaded!', 'danger')
            return redirect(url_for('main.predict'))

        face = request.files['face']
        iris = request.files['iris']
        finger = request.files['finger']

        if face and iris and finger and allowed_file(face.filename) and allowed_file(iris.filename) and allowed_file(finger.filename):
            face_path = os.path.join(UPLOAD_FOLDER, secure_filename(face.filename))
            iris_path = os.path.join(UPLOAD_FOLDER, secure_filename(iris.filename))
            finger_path = os.path.join(UPLOAD_FOLDER, secure_filename(finger.filename))

            face.save(face_path)
            iris.save(iris_path)
            finger.save(finger_path)

            authorized = predict_user(face_path, iris_path, fingerprint_path=finger_path)

            # Simulate match percentage (later you can use real value)
            authorized, user_label, match_percentage = predict_user(face_path, iris_path, fingerprint_path=finger_path)

            return render_template('result.html', authorized=authorized, user=user_label, match_percentage=match_percentage)

        else:
            flash('Invalid file format! Please upload JPG or PNG.', 'danger')
            return redirect(url_for('main.predict'))

    return render_template('predict.html')
