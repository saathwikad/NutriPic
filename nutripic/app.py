from flask import Flask, render_template, redirect, url_for, flash, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from wtforms import TextAreaField
import numpy as np


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder for uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

# Forms
class SignupForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=150)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email is already in use. Please choose a different one.')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Log In')

class WriteArticleForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired(), Length(max=150)])  # Article title
    content = TextAreaField('Content', validators=[DataRequired()])  # Article content
    submit = SubmitField('Publish Article')

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Routes
@app.route('/')
def landing_page():
    return render_template('index.html')  # Render the landing page template

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not authenticated
    return render_template('home.html')  # Render home page

# Article model
class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

@app.route('/write_articles', methods=['GET', 'POST'])
def write_articles():
    form = WriteArticleForm()  # Create an instance of the form
    if form.validate_on_submit():  # If form is submitted and valid
        # Create a new article and add it to the database
        article = Article(title=form.title.data, content=form.content.data)
        db.session.add(article)
        db.session.commit()  # Save the article to the database
        flash('Article published successfully!', 'success')
        return redirect(url_for('read_articles'))  # Redirect to the Read Articles page
    return render_template('write_articles.html', form=form)


@app.route('/read_articles')
def read_articles():
    articles = Article.query.all()  # Fetch all articles from the database
    return render_template('read_articles.html', articles=articles)

@app.route('/article/<int:article_id>')
def view_article(article_id):
    article = Article.query.get_or_404(article_id)
    return render_template('article.html', article=article)


# Routes for individual articles
@app.route('/article1')
def article1():
    return render_template('article1.html', title="The Wonders of Indian Cuisine", content="Indian cuisine is known for its vibrant spices and rich flavors...")

@app.route('/article2')
def article2():
    return render_template('article2.html', title="Healthy Eating Habits", content="Maintaining a healthy diet is essential for overall well-being...")

@app.route('/article3')
def article3():
    return render_template('article3.html', title="The Science of Food", content="Food plays a critical role in shaping our health and energy levels...")



@app.route('/ml_model_integration', methods=['GET', 'POST'])
def ml_model_integration():
    results = []
    components = {}
    
    if request.method == 'POST':
        file = request.files['image']
        
        if file and allowed_file(file.filename):
            # Save the file to the upload folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Load the InceptionV3 model pre-trained on ImageNet
            model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=True)

            # Preprocess the image for the model
            img = image.load_img(filename, target_size=(299, 299))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

            # Get model predictions
            predictions = model.predict(img_array)
            decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=3)[0]

            # Prepare mock nutritional data for demonstration (based on predictions)
            components = {
                "Carbohydrates": 50,  # Mock value (replace with actual logic from your model)
                "Proteins": 30,       # Mock value
                "Fats": 20            # Mock value
            }

            # Prepare results to be displayed
            for imagenet_id, label, score in decoded_predictions:
                results.append(f"{label}: {score:.2f}")
        else:
            flash('Invalid file format. Please upload a valid image (PNG, JPG, JPEG).', 'danger')

    # Convert the dictionary keys and values to lists for chart and table
    labels = list(components.keys())
    values = list(components.values())

    return render_template('ml_model_integration.html', results=results, components=components, labels=labels, values=values)

@app.route('/health_related')
def health_related():
    return render_template('health_related.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            session['user_id'] = user.id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
            return render_template('login.html', form=form, login_failed=True)
    return render_template('login.html', form=form, login_failed=False)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  # Create the upload folder if it doesn't exist
    with app.app_context():
        db.create_all()  # Ensure the database is created
    app.run()
