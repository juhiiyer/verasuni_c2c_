# Verasuni Authentication Backend
# Team: Hackstreet Boys ft. Gaurav
# Professional Login/Registration System with Google OAuth

from flask import Flask, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt
import datetime
import os
import re
from functools import wraps
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import uuid
from enum import Enum

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-super-secret-key-here'  # Change in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///verasuni.db'  # Use PostgreSQL in production
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads/documents'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Google OAuth Configuration
app.config['GOOGLE_CLIENT_ID'] = 'your-google-client-id'  # Get from Google Cloud Console
app.config['GOOGLE_CLIENT_SECRET'] = 'your-google-client-secret'

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# User Status Enum
class UserStatus(str, Enum):
    STUDENT = "student"
    ALUMNI = "alumni"
    FACULTY = "faculty"
    PARENT = "parent"
    UNVERIFIED = "unverified"

class DocumentType(str, Enum):
    STUDENT_ID = "student_id"
    DEGREE_CERTIFICATE = "degree_certificate"
    ENROLLMENT_CERTIFICATE = "enrollment_certificate"
    EMPLOYEE_ID = "employee_id"
    ALUMNI_ID = "alumni_id"
    OTHER = "other"

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Basic Info
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone_number = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)  # Nullable for Google OAuth users
    
    # User Status
    user_status = db.Column(db.Enum(UserStatus), nullable=False, default=UserStatus.UNVERIFIED)
    
    # College Information
    college_name = db.Column(db.String(200), nullable=True)
    course_branch = db.Column(db.String(100), nullable=True)
    graduation_year = db.Column(db.Integer, nullable=True)
    
    # Verification
    is_email_verified = db.Column(db.Boolean, default=False)
    is_document_verified = db.Column(db.Boolean, default=False)
    email_verification_token = db.Column(db.String(100), nullable=True)
    
    # OAuth
    google_id = db.Column(db.String(100), unique=True, nullable=True)
    oauth_provider = db.Column(db.String(50), nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    
    # Profile completion
    profile_completion = db.Column(db.Integer, default=30)  # Percentage
    
    # Relationships
    documents = db.relationship('UserDocument', backref='user', lazy=True, cascade='all, delete-orphan')
    social_profiles = db.relationship('SocialProfile', backref='user', lazy=True, cascade='all, delete-orphan')

class UserDocument(db.Model):
    __tablename__ = 'user_documents'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.user_id'), nullable=False)
    
    document_type = db.Column(db.Enum(DocumentType), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    mime_type = db.Column(db.String(100), nullable=False)
    
    is_verified = db.Column(db.Boolean, default=False)
    verification_notes = db.Column(db.Text, nullable=True)
    
    uploaded_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    verified_at = db.Column(db.DateTime, nullable=True)

class SocialProfile(db.Model):
    __tablename__ = 'social_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.user_id'), nullable=False)
    
    platform = db.Column(db.String(50), nullable=False)  # linkedin, facebook, twitter
    profile_url = db.Column(db.String(500), nullable=False)
    profile_name = db.Column(db.String(100), nullable=True)
    
    is_verified = db.Column(db.Boolean, default=False)
    added_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Utility Functions
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token.split(' ')[1]
            
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.filter_by(user_id=data['user_id']).first()
            
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

def generate_jwt_token(user):
    """Generate JWT token for authenticated user"""
    payload = {
        'user_id': user.user_id,
        'email': user.email,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)  # Token valid for 7 days
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_phone(phone):
    """Validate phone number format"""
    pattern = r'^[+]?[1-9]?[0-9]{7,15}$'
    return re.match(pattern, phone) is not None

def calculate_profile_completion(user):
    """Calculate user profile completion percentage"""
    completion = 0
    total_fields = 10
    
    if user.full_name: completion += 1
    if user.email: completion += 1
    if user.phone_number: completion += 1
    if user.user_status != UserStatus.UNVERIFIED: completion += 1
    if user.college_name: completion += 1
    if user.course_branch: completion += 1
    if user.graduation_year: completion += 1
    if user.is_email_verified: completion += 1
    if user.documents: completion += 1
    if user.social_profiles: completion += 1
    
    return int((completion / total_fields) * 100)

def allowed_file(filename):
    """Check if uploaded file is allowed"""
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes

@app.route('/')
def home():
    return jsonify({
        'message': 'Verasuni Authentication API',
        'version': '1.0.0',
        'endpoints': {
            'register': '/api/auth/register',
            'login': '/api/auth/login',
            'google_login': '/api/auth/google',
            'profile': '/api/user/profile',
            'upload_document': '/api/user/upload-document'
        }
    })

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['full_name', 'email', 'password', 'confirm_password', 'user_status']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Validate email format
        if not is_valid_email(data['email']):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=data['email'].lower()).first()
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 400
        
        # Validate password
        if len(data['password']) < 8:
            return jsonify({'error': 'Password must be at least 8 characters long'}), 400
        
        if data['password'] != data['confirm_password']:
            return jsonify({'error': 'Passwords do not match'}), 400
        
        # Validate phone number if provided
        if data.get('phone_number') and not is_valid_phone(data['phone_number']):
            return jsonify({'error': 'Invalid phone number format'}), 400
        
        # Validate user status
        if data['user_status'] not in [status.value for status in UserStatus]:
            return jsonify({'error': 'Invalid user status'}), 400
        
        # Create new user
        new_user = User(
            full_name=data['full_name'].strip(),
            email=data['email'].lower().strip(),
            phone_number=data.get('phone_number', '').strip() or None,
            password_hash=generate_password_hash(data['password']),
            user_status=UserStatus(data['user_status']),
            college_name=data.get('college_name', '').strip() or None,
            course_branch=data.get('course_branch', '').strip() or None,
            graduation_year=data.get('graduation_year') if data.get('graduation_year') else None,
            email_verification_token=str(uuid.uuid4())
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        # Calculate profile completion
        new_user.profile_completion = calculate_profile_completion(new_user)
        db.session.commit()
        
        # Generate JWT token
        token = generate_jwt_token(new_user)
        
        return jsonify({
            'message': 'Registration successful',
            'user': {
                'user_id': new_user.user_id,
                'full_name': new_user.full_name,
                'email': new_user.email,
                'user_status': new_user.user_status.value,
                'profile_completion': new_user.profile_completion
            },
            'token': token
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user by email
        user = User.query.filter_by(email=data['email'].lower().strip()).first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Update last login
        user.last_login = datetime.datetime.utcnow()
        db.session.commit()
        
        # Generate JWT token
        token = generate_jwt_token(user)
        
        return jsonify({
            'message': 'Login successful',
            'user': {
                'user_id': user.user_id,
                'full_name': user.full_name,
                'email': user.email,
                'user_status': user.user_status.value,
                'profile_completion': user.profile_completion,
                'is_email_verified': user.is_email_verified,
                'is_document_verified': user.is_document_verified
            },
            'token': token
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/auth/google', methods=['POST'])
def google_login():
    """Google OAuth login endpoint"""
    try:
        data = request.get_json()
        google_token = data.get('token')
        
        if not google_token:
            return jsonify({'error': 'Google token is required'}), 400
        
        # Verify Google token
        try:
            idinfo = id_token.verify_oauth2_token(
                google_token, 
                google_requests.Request(), 
                app.config['GOOGLE_CLIENT_ID']
            )
            
            google_id = idinfo['sub']
            email = idinfo['email']
            name = idinfo.get('name', '')
            
        except ValueError as e:
            return jsonify({'error': 'Invalid Google token'}), 400
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Update Google ID if not set
            if not user.google_id:
                user.google_id = google_id
                user.oauth_provider = 'google'
            
            # Update last login
            user.last_login = datetime.datetime.utcnow()
            db.session.commit()
            
        else:
            # Create new user
            user = User(
                full_name=name,
                email=email,
                google_id=google_id,
                oauth_provider='google',
                is_email_verified=True,  # Google emails are pre-verified
                user_status=UserStatus.UNVERIFIED
            )
            
            db.session.add(user)
            db.session.commit()
            
            # Calculate profile completion
            user.profile_completion = calculate_profile_completion(user)
            db.session.commit()
        
        # Generate JWT token
        token = generate_jwt_token(user)
        
        return jsonify({
            'message': 'Google login successful',
            'user': {
                'user_id': user.user_id,
                'full_name': user.full_name,
                'email': user.email,
                'user_status': user.user_status.value,
                'profile_completion': user.profile_completion,
                'is_email_verified': user.is_email_verified,
                'is_document_verified': user.is_document_verified
            },
            'token': token,
            'is_new_user': user.profile_completion < 50
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Google login failed: {str(e)}'}), 500

@app.route('/api/user/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    """Get user profile"""
    try:
        documents = [{
            'id': doc.id,
            'document_type': doc.document_type.value,
            'file_name': doc.file_name,
            'file_size': doc.file_size,
            'is_verified': doc.is_verified,
            'uploaded_at': doc.uploaded_at.isoformat()
        } for doc in current_user.documents]
        
        social_profiles = [{
            'id': social.id,
            'platform': social.platform,
            'profile_url': social.profile_url,
            'profile_name': social.profile_name,
            'is_verified': social.is_verified
        } for social in current_user.social_profiles]
        
        return jsonify({
            'user': {
                'user_id': current_user.user_id,
                'full_name': current_user.full_name,
                'email': current_user.email,
                'phone_number': current_user.phone_number,
                'user_status': current_user.user_status.value,
                'college_name': current_user.college_name,
                'course_branch': current_user.course_branch,
                'graduation_year': current_user.graduation_year,
                'profile_completion': current_user.profile_completion,
                'is_email_verified': current_user.is_email_verified,
                'is_document_verified': current_user.is_document_verified,
                'oauth_provider': current_user.oauth_provider,
                'created_at': current_user.created_at.isoformat(),
                'last_login': current_user.last_login.isoformat() if current_user.last_login else None
            },
            'documents': documents,
            'social_profiles': social_profiles
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch profile: {str(e)}'}), 500

@app.route('/api/user/profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    """Update user profile"""
    try:
        data = request.get_json()
        
        # Update allowed fields
        if data.get('full_name'):
            current_user.full_name = data['full_name'].strip()
        
        if data.get('phone_number'):
            if not is_valid_phone(data['phone_number']):
                return jsonify({'error': 'Invalid phone number format'}), 400
            current_user.phone_number = data['phone_number'].strip()
        
        if data.get('user_status') and data['user_status'] in [status.value for status in UserStatus]:
            current_user.user_status = UserStatus(data['user_status'])
        
        if data.get('college_name'):
            current_user.college_name = data['college_name'].strip()
        
        if data.get('course_branch'):
            current_user.course_branch = data['course_branch'].strip()
        
        if data.get('graduation_year'):
            current_user.graduation_year = data['graduation_year']
        
        # Recalculate profile completion
        current_user.profile_completion = calculate_profile_completion(current_user)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Profile updated successfully',
            'profile_completion': current_user.profile_completion
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Profile update failed: {str(e)}'}), 500

@app.route('/api/user/upload-document', methods=['POST'])
@token_required
def upload_document(current_user):
    """Upload verification document"""
    try:
        if 'document' not in request.files:
            return jsonify({'error': 'No document file provided'}), 400
        
        file = request.files['document']
        document_type = request.form.get('document_type')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not document_type or document_type not in [doc_type.value for doc_type in DocumentType]:
            return jsonify({'error': 'Invalid document type'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate secure filename
        filename = secure_filename(f"{current_user.user_id}_{document_type}_{int(datetime.datetime.utcnow().timestamp())}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        # Save document record
        document = UserDocument(
            user_id=current_user.user_id,
            document_type=DocumentType(document_type),
            file_name=file.filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=file.content_type or 'application/octet-stream'
        )
        
        db.session.add(document)
        
        # Update profile completion
        current_user.profile_completion = calculate_profile_completion(current_user)
        
        db.session.commit()
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'document': {
                'id': document.id,
                'document_type': document.document_type.value,
                'file_name': document.file_name,
                'file_size': document.file_size,
                'uploaded_at': document.uploaded_at.isoformat()
            },
            'profile_completion': current_user.profile_completion
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Document upload failed: {str(e)}'}), 500

@app.route('/api/user/add-social-profile', methods=['POST'])
@token_required
def add_social_profile(current_user):
    """Add social media profile"""
    try:
        data = request.get_json()
        
        platform = data.get('platform', '').lower().strip()
        profile_url = data.get('profile_url', '').strip()
        profile_name = data.get('profile_name', '').strip()
        
        if not platform or not profile_url:
            return jsonify({'error': 'Platform and profile URL are required'}), 400
        
        if platform not in ['linkedin', 'facebook', 'twitter', 'instagram']:
            return jsonify({'error': 'Invalid platform'}), 400
        
        # Check if profile already exists
        existing_profile = SocialProfile.query.filter_by(
            user_id=current_user.user_id, 
            platform=platform
        ).first()
        
        if existing_profile:
            return jsonify({'error': f'{platform.title()} profile already added'}), 400
        
        # Add social profile
        social_profile = SocialProfile(
            user_id=current_user.user_id,
            platform=platform,
            profile_url=profile_url,
            profile_name=profile_name
        )
        
        db.session.add(social_profile)
        
        # Update profile completion
        current_user.profile_completion = calculate_profile_completion(current_user)
        
        db.session.commit()
        
        return jsonify({
            'message': f'{platform.title()} profile added successfully',
            'profile_completion': current_user.profile_completion
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to add social profile: {str(e)}'}), 500

@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout(current_user):
    """User logout endpoint"""
    # Note: With JWT, logout is mainly handled on client-side by removing the token
    # For additional security, you could maintain a token blacklist
    return jsonify({'message': 'Logout successful'}), 200

# Admin Routes (for testing/management)
@app.route('/api/admin/users', methods=['GET'])
def list_users():
    """List all users (admin endpoint for testing)"""
    try:
        users = User.query.all()
        users_list = []
        
        for user in users:
            users_list.append({
                'user_id': user.user_id,
                'full_name': user.full_name,
                'email': user.email,
                'user_status': user.user_status.value,
                'profile_completion': user.profile_completion,
                'is_email_verified': user.is_email_verified,
                'is_document_verified': user.is_document_verified,
                'created_at': user.created_at.isoformat()
            })
        
        return jsonify({
            'users': users_list,
            'total_count': len(users_list)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch users: {str(e)}'}), 500

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

# Initialize Database
def init_db():
    """Initialize database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    init_db()
    print("Starting Verasuni Authentication Server...")
    print("Available endpoints:")
    print("- POST /api/auth/register - User registration")
    print("- POST /api/auth/login - User login") 
    print("- POST /api/auth/google - Google OAuth login")
    print("- GET /api/user/profile - Get user profile")
    print("- PUT /api/user/profile - Update user profile")
    print("- POST /api/user/upload-document - Upload verification document")
    print("- POST /api/user/add-social-profile - Add social media profile")
    print("- POST /api/auth/logout - User logout")
    
    app.run(debug=True, host='0.0.0.0', port=5000)