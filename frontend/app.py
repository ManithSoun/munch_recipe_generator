from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
import secrets
from matching_algorithm import find_recipe, find_recipe_from_image
import csv
import datetime

app = Flask(__name__)
SECRET_KEY_FILE = 'secret_key.txt'

# Secret key generation
if os.path.exists(SECRET_KEY_FILE):
    with open(SECRET_KEY_FILE, 'r') as f:
        app.secret_key = f.read().strip()
else:
    secret = secrets.token_hex(16)
    with open(SECRET_KEY_FILE, 'w') as f:
        f.write(secret)
    app.secret_key = secret

# Upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load dataset
try:
    df = pd.read_csv('recipes.csv')
    print("Columns in dataset:", df.columns)
    if 'Instruction' in df.columns:
        df.drop_duplicates(subset=['Instruction'], inplace=True)
        df.dropna(subset=['Instruction'], inplace=True)
    else:
        raise KeyError("Column 'Instruction' not found in dataset.")
except FileNotFoundError:
    raise Exception("Dataset file 'recipes.csv' not found.")
except KeyError as e:
    raise Exception(f"Missing required column: {str(e)}")

# Check file extension
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    generated_recipe = ""
    uploaded_image_path = ""
    text_input = ""
    error_message = ""

    if request.method == 'POST':
        text_input = request.form.get('text_input', "").strip()
        image = request.files.get('image')

        if text_input and image and image.filename:
            error_message = "Please provide either text input or an image, not both."
        elif text_input:
            generated_recipe = find_recipe(text_input)
        elif image and image.filename:
            if not allowed_file(image.filename):
                error_message = "Invalid file type. Please upload an image file."
            else:
                filename = secure_filename(image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    image.save(image_path)
                    uploaded_image_path = filename
                    generated_recipe = find_recipe_from_image(image_path)
                except Exception as e:
                    error_message = f"Error saving or predicting: {str(e)}"
        else:
            error_message = "Please provide either ingredients or an image."

        if error_message:
            flash(error_message)

    return render_template(
        'index.html',
        user_input=text_input,
        generated_recipe=generated_recipe,
        uploaded_image=uploaded_image_path
    )

@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    input_data = request.form.get('user_input', '')
    generated_recipe = request.form.get('generated_recipe', '')
    rating = request.form.get('rating', '')
    comment = request.form.get('comment', '')

    # Save feedback
    with open('feedback.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            input_data,
            generated_recipe,
            rating,
            comment,
            datetime.datetime.now().isoformat()
        ])

    flash('Thank you for your feedback!')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
