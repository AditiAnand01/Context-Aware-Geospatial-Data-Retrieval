from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate

from forms import LoginForm, RegistrationForm
from Preprocess_query.main import generate_output

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_api_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///isro_data.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database model for User
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    chats = db.relationship('Chat', backref='user', lazy=True)

# Database model for Chat
class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sender = db.Column(db.String(50), nullable=False)  # New column to indicate the sender ('user' or 'bot')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
@login_required
def index():
    user_chats = Chat.query.filter_by(user_id=current_user.id).all()
    return render_template('index.html', chats=user_chats)

# Route to serve the index.html page
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()  # Create an instance of the RegistrationForm
    if form.validate_on_submit():  # Check if the form is submitted and valid
        username = form.username.data
        password = form.password.data
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)  # Pass the form to the template

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()  # Create an instance of the LoginForm
    if form.validate_on_submit():  # Check if the form is submitted and valid
        username = form.username.data
        password = form.password.data
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', form=form)  # Pass the form to the template

# Route for user dashboard
@app.route('/dashboard')
@login_required
def dashboard():
    user_chats = Chat.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', chats=user_chats)

# Route to handle processing the user query
@app.route('/process_query', methods=['POST'])
@login_required
def process_query():
    data = request.json
    user_query = data.get('query', '')
    bot_response = generate_output(user_query)

    # Save user query
    user_chat = Chat(content=user_query, user_id=current_user.id, sender='user')
    db.session.add(user_chat)
    
    # Save bot response
    bot_chat = Chat(content=bot_response, user_id=current_user.id, sender='bot')
    db.session.add(bot_chat)
    
    db.session.commit()
    
    return jsonify({'response': bot_response})

# Route for user logout
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
