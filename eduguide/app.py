import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure random key

# ========================
# Database Configuration
# ========================
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Update with your MySQL password
    'database': 'ml_dashboard'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
        return None

# ========================
# Load Models and Encoders (Model 1: Stream/School)
# ========================
try:
    stream_model = joblib.load("models/streams_model.pkl")
    target_encoder = joblib.load("models/target_encoder.pkl")
    interest_encoder = joblib.load("models/interest_encoder.pkl")
except FileNotFoundError as e:
    print(f"Model file missing: {e}")
    exit(1)

# ========================
# Load Models and Encoders (Model 2: Degree/College)
# ========================
try:
    degree_model = joblib.load("models/best_model.pkl")
    gender_encoder = joblib.load("models/gender_encoder.pkl")
    stream_encoder = joblib.load("models/stream_encoder.pkl")
    percentage_encoder = joblib.load("models/percentage_encoder.pkl")
    location_encoder = joblib.load("models/location_encoder.pkl")
    parent_occupation_encoder = joblib.load("models/parent_occupation_encoder.pkl")
    career_interest_encoder = joblib.load("models/career_interest_encoder.pkl")
    preferred_degree_encoder = joblib.load("models/preferred_degree_encoder.pkl")
except FileNotFoundError as e:
    print(f"Model file missing: {e}")
    exit(1)

# Get the distinct percentage values the encoder was trained on (as floats)
try:
    train_percentages = percentage_encoder.classes_.astype(float)
except AttributeError:
    train_percentages = np.array([0, 100])  # fallback

# ========================
# Load External Datasets for Recommendations
# ========================
try:
    school_data = pd.read_csv("data/school data.csv")
    college_data = pd.read_csv("data/colleges3.csv")
except FileNotFoundError as e:
    print(f"Data file missing: {e}")
    exit(1)

# Normalize college degree column for filtering
college_data['Preferred_Degree'] = college_data['Preferred_Degree'].astype(str).str.strip().str.lower()

# ========================
# Helper Functions
# ========================
def encode_interest(interest_name):
    """Map interest name to its encoded value (Model 1)."""
    try:
        return interest_encoder.transform([interest_name])[0]
    except ValueError:
        flash(f"Interest '{interest_name}' not recognized. Using default.", "warning")
        return 0

def encode_percentage(percentage):
    """
    Map a continuous percentage to the closest value that the LabelEncoder was trained on.
    """
    if percentage in train_percentages:
        return percentage_encoder.transform([percentage])[0]
    else:
        closest = min(train_percentages, key=lambda x: abs(x - percentage))
        flash(f"Percentage {percentage} not in training data. Using closest value {closest}.", "info")
        return percentage_encoder.transform([closest])[0]

# ========================
# Routes
# ========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/predict_sslc')
def predict_sslc():
    if 'user_id' not in session:
        flash("Please login to access this page.", "warning")
        return redirect(url_for('login'))
    return render_template('predict_sslc.html')

@app.route('/predict_hsc')
def predict_hsc():
    if 'user_id' not in session:
        flash("Please login to access this page.", "warning")
        return redirect(url_for('login'))
    return render_template('predict_hsc.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        hashed_pw = generate_password_hash(password)

        conn = get_db_connection()
        if conn is None:
            flash("Database connection error", "error")
            return render_template('register.html')

        cursor = conn.cursor()

        try:
            sql = """
                INSERT INTO users (username, email, password)
                VALUES (%s, %s, %s)
            """
            values = (username, email, hashed_pw)
            cursor.execute(sql, values)
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))

        except mysql.connector.IntegrityError:
            flash("Username or email already exists", "error")
        except Error as e:
            print(e)
            flash("Database error occurred", "error")
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        if not conn:
            flash("Database connection error", "error")
            return render_template('login.html')

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash("Logged in successfully.", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.", "error")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for('index'))

# ========================
# Prediction Endpoints (AJAX)
# ========================
@app.route('/predict_school', methods=['POST'])
def predict_school():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})

    try:
        maths = int(request.form['maths'])
        science = int(request.form['science'])
        social = int(request.form['social'])
        english = int(request.form['english'])
        interest_name = request.form['interest']

        for mark in [maths, science, social, english]:
            if mark < 35 or mark > 100:
                return jsonify({'success': False, 'error': 'Marks must be between 35 and 100.'})

        interest_encoded = encode_interest(interest_name)

        input_data = [[maths, science, social, english, interest_encoded]]
        pred_encoded = stream_model.predict(input_data)[0]
        predicted_group = target_encoder.inverse_transform([pred_encoded])[0]

        eligible = school_data[
            (school_data['course'].str.contains(predicted_group, case=False, na=False)) &
            (school_data['min_maths'] <= maths) &
            (school_data['min_science'] <= science)
        ]
        schools = eligible[['school_name', 'school_board', 'district', 'annual_fee', 'ranking']].to_dict('records')

        return jsonify({
            'success': True,
            'group': predicted_group,
            'schools': schools
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_college', methods=['POST'])
def predict_college():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})

    try:
        gender = request.form['gender']
        stream = request.form['stream']
        percentage = float(request.form['percentage'])
        location = request.form['location']
        parent_occupation = request.form['parent_occupation']
        career_interest = request.form['career_interest']

        if percentage < 0 or percentage > 100:
            return jsonify({'success': False, 'error': 'Percentage must be between 0 and 100.'})

        gender_enc = gender_encoder.transform([gender])[0]
        stream_enc = stream_encoder.transform([stream])[0]
        percentage_enc = encode_percentage(percentage)
        location_enc = location_encoder.transform([location])[0]
        parent_enc = parent_occupation_encoder.transform([parent_occupation])[0]
        career_enc = career_interest_encoder.transform([career_interest])[0]

        input_data = [[gender_enc, stream_enc, percentage_enc, location_enc, parent_enc, career_enc]]
        pred_encoded = degree_model.predict(input_data)[0]
        predicted_degree = preferred_degree_encoder.inverse_transform([pred_encoded])[0]

        colleges = college_data[college_data['Preferred_Degree'] == predicted_degree.lower()]
        colleges_list = colleges[['College', 'City']].to_dict('records')

        return jsonify({
            'success': True,
            'degree': predicted_degree,
            'colleges': colleges_list
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
