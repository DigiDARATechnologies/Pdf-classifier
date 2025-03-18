from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template, flash, session
import os
from functools import wraps
from pdf_classifier import classify_pdf, PDFClassifierANN  # Import the model class
from database import get_db_connection
from datetime import datetime
import torch
import joblib

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model, vectorizer, and label_map at startup with proper model instantiation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'project', 'models')
print(f"Base directory: {BASE_DIR}")
print(f"Model directory: {MODEL_DIR}")

try:
    model_path = os.path.join(MODEL_DIR, 'ann_model.pth')
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    label_map_path = os.path.join(MODEL_DIR, 'label_map.pkl')
    print(f"Attempting to load model from: {model_path}")
    print(f"Attempting to load vectorizer from: {vectorizer_path}")
    print(f"Attempting to load label_map from: {label_map_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"Vectorizer file exists: {os.path.exists(vectorizer_path)}")
    print(f"Label map file exists: {os.path.exists(label_map_path)}")

    # Load vectorizer and label_map first to get input_size and num_classes
    vectorizer = joblib.load(vectorizer_path)
    label_map = joblib.load(label_map_path)
    num_classes = len(label_map)
    input_size = vectorizer.transform(['dummy text']).shape[1]  # Dynamically get input size

    # Instantiate the model
    model = PDFClassifierANN(input_size, hidden_size=128, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model, vectorizer, and label_map loaded successfully")
except Exception as e:
    print(f"Error loading model dependencies: {e}")
    model, vectorizer, label_map = None, None, None

# Hardcoded users
users = {
    'admin': {'password': 'password123', 'role': 'admin'},
    'staff': {'password': 'password123', 'role': 'staff'},
    'student': {'password': 'password123', 'role': 'student'}
}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users.get(username)
        if user and user['password'] == password:
            session['username'] = username
            session['role'] = user['role']
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user['role'] == 'staff':
                return redirect(url_for('staff_dashboard'))
            elif user['role'] == 'student':
                return redirect(url_for('student_dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin_dashboard.html')

@app.route('/staff_dashboard')
@login_required
def staff_dashboard():
    if session.get('role') != 'staff':
        return redirect(url_for('login'))
    return render_template('staff_dashboard.html')

@app.route('/student_dashboard')
@login_required
def student_dashboard():
    if session.get('role') != 'student':
        return redirect(url_for('login'))
    return render_template('student_dashboard.html')

@app.route('/exam')
@login_required
def exam():
    return render_template('exam.html')

@app.route('/student')
@login_required
def student():
    return render_template('student.html')

@app.route('/download_attachments')
@login_required
def download_attachments():
    return "Download Attachments Page"

@app.route('/upload', methods=['POST'])
@login_required
def upload_pdf():
    if model is None or vectorizer is None or label_map is None:
        flash('Model dependencies not loaded. Please check server logs.')
        return redirect(url_for('admin_dashboard'))

    if 'pdf' not in request.files:
        flash('No file part')
        return redirect(url_for('admin_dashboard'))

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        flash('No selected file')
        return redirect(url_for('admin_dashboard'))

    if not pdf_file.filename.endswith('.pdf'):
        flash('File is not a PDF')
        return redirect(url_for('admin_dashboard'))

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    pdf_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
    pdf_file.save(pdf_file_path)

    # Classify the PDF with loaded dependencies
    category = classify_pdf(pdf_file_path, vectorizer, model, label_map)

    conn = get_db_connection()
    if conn is None:
        flash('Database connection failed')
        return redirect(url_for('admin_dashboard'))
    cursor = conn.cursor()
    cursor.execute('INSERT INTO uploads (filename, category, uploaded_by, upload_date) VALUES (%s, %s, %s, %s)',
                   (pdf_file.filename, category, session['username'], datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    cursor.close()
    conn.close()

    flash('File uploaded and classified as ' + category)
    return redirect(url_for('admin_dashboard'))

@app.route('/pdfs/<category>', methods=['GET'])
@login_required
def list_pdfs_by_category(category):
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM uploads WHERE category = %s', (category,))
    pdfs = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(pdfs), 200

@app.route('/download/<filename>', methods=['GET'])
@login_required
def download_pdf(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)