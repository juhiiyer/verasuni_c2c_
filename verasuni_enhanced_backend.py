# Verasuni Complete Backend with Anonymous Reviews and Web Scraping
# Team: Hackstreet Boys ft. Gaurav
# Updated Authentication Backend with Review System

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
import json
from bs4 import BeautifulSoup
import time
import random

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-super-secret-key-here'  
# Update to use MySQL (pymysql driver)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:mypassword@localhost/verasuni_db'  # Update with your actual MySQL username, password, and database name
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads/documents'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Google OAuth Configuration
app.config['GOOGLE_CLIENT_ID'] = 'your-google-client-id'  
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

class ReviewCategory(str, Enum):
    ACADEMICS = "academics"
    CAMPUS_LIFE = "campus_life"
    PLACEMENTS = "placements"
    FACILITIES = "facilities"
    FOOD = "food"
    HOSTEL = "hostel"
    FACULTY = "faculty"
    OVERALL = "overall"

# Database Models
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Basic Info
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    phone_number = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)
    
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
    profile_completion = db.Column(db.Integer, default=30)
    
    # Relationships
    documents = db.relationship('UserDocument', backref='user', lazy=True, cascade='all, delete-orphan')
    social_profiles = db.relationship('SocialProfile', backref='user', lazy=True, cascade='all, delete-orphan')
    reviews = db.relationship('Review', backref='user', lazy=True, cascade='all, delete-orphan')

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
    
    platform = db.Column(db.String(50), nullable=False)
    profile_url = db.Column(db.String(500), nullable=False)
    profile_name = db.Column(db.String(100), nullable=True)
    
    is_verified = db.Column(db.Boolean, default=False)
    added_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# NEW: Review Model with Anonymous Support
class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.user_id'), nullable=False)
    
    # Review Content
    content = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=True)  # Overall rating (1-5)
    category = db.Column(db.Enum(ReviewCategory), default=ReviewCategory.OVERALL)
    
    # College Information
    college_name = db.Column(db.String(200), nullable=False)
    course_branch = db.Column(db.String(100), nullable=True)
    
    # Privacy Settings - NEW FEATURE
    is_anonymous = db.Column(db.Boolean, default=False)  # Anonymous posting
    
    # Source tracking
    source = db.Column(db.String(50), default='user_generated')  # 'user_generated', 'web_scraped', 'official'
    source_url = db.Column(db.String(500), nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Bias Analysis Results (stored after processing)
    bias_score = db.Column(db.Float, nullable=True)
    corrected_content = db.Column(db.Text, nullable=True)
    credibility_score = db.Column(db.Float, nullable=True)

class College(db.Model):
    __tablename__ = 'colleges'
    
    id = db.Column(db.Integer, primary_key=True)
    college_id = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    name = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200), nullable=True)
    website = db.Column(db.String(500), nullable=True)
    
    # Official Data
    nirf_ranking = db.Column(db.Integer, nullable=True)
    established_year = db.Column(db.Integer, nullable=True)
    total_students = db.Column(db.Integer, nullable=True)
    
    # Scraped Data Metadata
    last_scraped = db.Column(db.DateTime, nullable=True)
    scraping_enabled = db.Column(db.Boolean, default=True)
    
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

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
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
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

# Web Scraping Functions - NEW FEATURE
def fetch_official_reviews(college_name, limit=20):
    """
    Fetch official reviews from web sources when site has few reviews
    This is called automatically when site reviews < 50
    """
    try:
        official_reviews = []
        
        # Source 1: CollegeDunia (simulated - replace with actual scraping)
        collegedunia_reviews = scrape_collegedunia(college_name, limit//2)
        official_reviews.extend(collegedunia_reviews)
        
        # Source 2: Shiksha.com (simulated - replace with actual scraping)
        shiksha_reviews = scrape_shiksha(college_name, limit//2)
        official_reviews.extend(shiksha_reviews)
        
        return official_reviews[:limit]
        
    except Exception as e:
        print(f"Error fetching official reviews: {e}")
        return []

def scrape_collegedunia(college_name, limit=10):
    """Scrape reviews from CollegeDunia"""
    try:
        # Simulate web scraping (replace with actual implementation)
        time.sleep(1)  # Respect rate limits
        
        # In real implementation, use requests + BeautifulSoup
        # url = f"https://collegedunia.com/college/{college_name.lower().replace(' ', '-')}/reviews"
        # response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0...'})
        # soup = BeautifulSoup(response.content, 'html.parser')
        
        # For demo, return simulated reviews
        sample_reviews = [
            {
                'content': f'Good college with decent placement opportunities. The faculty is experienced and campus infrastructure is well-maintained.',
                'rating': 4.2,
                'category': 'overall',
                'source': 'collegedunia',
                'source_url': f'https://collegedunia.com/college/{college_name.lower()}/reviews',
                'is_anonymous': True
            },
            {
                'content': f'The academic curriculum is updated and relevant to industry needs. Good lab facilities and library resources.',
                'rating': 4.0,
                'category': 'academics',
                'source': 'collegedunia',
                'source_url': f'https://collegedunia.com/college/{college_name.lower()}/reviews',
                'is_anonymous': True
            }
        ]
        
        return sample_reviews[:limit]
        
    except Exception as e:
        print(f"Error scraping CollegeDunia: {e}")
        return []

def scrape_shiksha(college_name, limit=10):
    """Scrape reviews from Shiksha.com"""
    try:
        # Simulate web scraping (replace with actual implementation)
        time.sleep(1)  # Respect rate limits
        
        sample_reviews = [
            {
                'content': f'Great campus life and active student communities. The placement cell is quite supportive.',
                'rating': 4.1,
                'category': 'campus_life',
                'source': 'shiksha',
                'source_url': f'https://shiksha.com/college/{college_name.lower()}-reviews',
                'is_anonymous': True
            },
            {
                'content': f'Hostel facilities are good with decent food quality. Wi-Fi connectivity is reliable across campus.',
                'rating': 3.8,
                'category': 'facilities',
                'source': 'shiksha', 
                'source_url': f'https://shiksha.com/college/{college_name.lower()}-reviews',
                'is_anonymous': True
            }
        ]
        
        return sample_reviews[:limit]
        
    except Exception as e:
        print(f"Error scraping Shiksha: {e}")
        return []

def check_review_count_and_scrape(college_name):
    """
    Check if we have enough reviews for a college.
    If not, automatically fetch official reviews.
    """
    MINIMUM_REVIEWS_THRESHOLD = 50
    
    current_review_count = Review.query.filter_by(
        college_name=college_name,
        is_active=True
    ).count()
    
    if current_review_count < MINIMUM_REVIEWS_THRESHOLD:
        print(f"Low review count ({current_review_count}) for {college_name}. Fetching official reviews...")
        
        # Fetch official reviews
        official_reviews = fetch_official_reviews(college_name, 
                                                 limit=MINIMUM_REVIEWS_THRESHOLD - current_review_count)
        
        # Store official reviews in database
        for review_data in official_reviews:
            try:
                # Create a system user for official reviews if not exists
                system_user = User.query.filter_by(email='system@verasuni.com').first()
                if not system_user:
                    system_user = User(
                        full_name='System',
                        email='system@verasuni.com',
                        user_status=UserStatus.UNVERIFIED
                    )
                    db.session.add(system_user)
                    db.session.commit()
                
                review = Review(
                    user_id=system_user.user_id,
                    content=review_data['content'],
                    rating=review_data.get('rating'),
                    category=ReviewCategory(review_data.get('category', 'overall')),
                    college_name=college_name,
                    is_anonymous=True,  # Official reviews are always anonymous
                    source=review_data.get('source', 'web_scraped'),
                    source_url=review_data.get('source_url')
                )
                
                db.session.add(review)
                
            except Exception as e:
                print(f"Error storing official review: {e}")
                continue
        
        db.session.commit()
        print(f"Added {len(official_reviews)} official reviews for {college_name}")
        
        return len(official_reviews)
    
    return 0

# Routes

@app.route('/')
def home():
    return jsonify({
        'message': 'Verasuni Authentication & Review API',
        'version': '2.0.0',
        'features': ['Anonymous Reviews', 'Web Scraping Fallback', 'Bias Detection'],
        'endpoints': {
            'auth': {
                'register': '/api/auth/register',
                'login': '/api/auth/login',
                'google_login': '/api/auth/google'
            },
            'user': {
                'profile': '/api/user/profile',
                'upload_document': '/api/user/upload-document'
            },
            'reviews': {
                'post_review': '/api/reviews',
                'get_reviews': '/api/reviews/<college_name>',
                'get_my_reviews': '/api/user/reviews'
            },
            'colleges': {
                'list_colleges': '/api/colleges',
                'get_college': '/api/colleges/<college_id>'
            }
        }
    })

# Authentication Routes (same as before)
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

# NEW: Review Routes with Anonymous Support
@app.route('/api/reviews', methods=['POST'])
@token_required
def post_review(current_user):
    """Post a new review with anonymous option"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['content', 'college_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400
        
        # Validate rating if provided
        if data.get('rating'):
            rating = float(data['rating'])
            if not (1.0 <= rating <= 5.0):
                return jsonify({'error': 'Rating must be between 1.0 and 5.0'}), 400
        
        # Validate category
        category = data.get('category', 'overall')
        if category not in [cat.value for cat in ReviewCategory]:
            return jsonify({'error': 'Invalid review category'}), 400
        
        # NEW: Handle anonymous posting
        is_anonymous = data.get('is_anonymous', False)
        
        # Create new review
        new_review = Review(
            user_id=current_user.user_id,
            content=data['content'].strip(),
            rating=data.get('rating'),
            category=ReviewCategory(category),
            college_name=data['college_name'].strip(),
            course_branch=data.get('course_branch', '').strip() or None,
            is_anonymous=is_anonymous,  # NEW: Anonymous flag
            source='user_generated'
        )
        
        db.session.add(new_review)
        db.session.commit()
        
        # After posting, check if we need to scrape more reviews for this college
        scraped_count = check_review_count_and_scrape(data['college_name'])
        
        response_data = {
            'message': 'Review posted successfully',
            'review': {
                'review_id': new_review.review_id,
                'content': new_review.content,
                'rating': new_review.rating,
                'category': new_review.category.value,
                'college_name': new_review.college_name,
                'is_anonymous': new_review.is_anonymous,
                'created_at': new_review.created_at.isoformat()
            }
        }
        
        if scraped_count > 0:
            response_data['info'] = f'Added {scraped_count} official reviews to improve recommendations'
        
        return jsonify(response_data), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to post review: {str(e)}'}), 500

@app.route('/api/reviews/<college_name>', methods=['GET'])
def get_reviews(college_name):
    """Get all reviews for a college (respects anonymous setting)"""
    try:
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        category = request.args.get('category')

        # Fetch reviews from MySQL database using SQLAlchemy ORM
        query = Review.query.filter_by(
            college_name=college_name,
            is_active=True
        )
        if category and category in [cat.value for cat in ReviewCategory]:
            query = query.filter_by(category=ReviewCategory(category))

        reviews = query.order_by(Review.created_at.desc()).offset(offset).limit(limit).all()

        reviews_data = []
        for review in reviews:
            review_data = {
                'review_id': review.review_id,
                'content': review.content,
                'rating': review.rating,
                'category': review.category.value,
                'college_name': review.college_name,
                'course_branch': review.course_branch,
                'created_at': review.created_at.isoformat(),
                'source': review.source
            }
            if review.is_anonymous:
                review_data['author'] = 'Anonymous'
                review_data['is_anonymous'] = True
            else:
                user = User.query.filter_by(user_id=review.user_id).first()
                if user:
                    review_data['author'] = user.full_name
                    review_data['author_status'] = user.user_status.value
                review_data['is_anonymous'] = False
            reviews_data.append(review_data)

        total_count = Review.query.filter_by(
            college_name=college_name,
            is_active=True
        ).count()

        return jsonify({
            'reviews': reviews_data,
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'college_name': college_name
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to fetch reviews: {str(e)}'}), 500

@app.route('/api/user/reviews', methods=['GET'])
@token_required
def get_user_reviews(current_user):
    """Get all reviews posted by current user"""
    try:
        reviews = Review.query.filter_by(
            user_id=current_user.user_id,
            is_active=True
        ).order_by(Review.created_at.desc()).all()
        
        reviews_data = []
        for review in reviews:
            review_data = {
                'review_id': review.review_id,
                'content': review.content,
                'rating': review.rating,
                'category': review.category.value,
                'college_name': review.college_name,
                'course_branch': review.course_branch,
                'is_anonymous': review.is_anonymous,
                'source': review.source,
                'created_at': review.created_at.isoformat()
            }
            
            # Include bias analysis results if available
            if review.bias_score is not None:
                review_data['bias_analysis'] = {
                    'bias_score': review.bias_score,
                    'corrected_content': review.corrected_content,
                    'credibility_score': review.credibility_score
                }
            
            reviews_data.append(review_data)
        
        return jsonify({
            'reviews': reviews_data,
            'total_count': len(reviews_data)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch user reviews: {str(e)}'}), 500

@app.route('/api/colleges', methods=['GET'])
def list_colleges():
    """List all colleges with review counts"""
    try:
        # Get colleges with review counts
        colleges_with_reviews = db.session.query(
            Review.college_name,
            db.func.count(Review.id).label('review_count'),
            db.func.avg(Review.rating).label('avg_rating')
        ).filter(
            Review.is_active == True
        ).group_by(Review.college_name).all()
        
        colleges_data = []
        for college in colleges_with_reviews:
            colleges_data.append({
                'college_name': college.college_name,
                'review_count': college.review_count,
                'average_rating': round(college.avg_rating, 2) if college.avg_rating else None
            })
        
        # Sort by review count
        colleges_data.sort(key=lambda x: x['review_count'], reverse=True)
        
        return jsonify({
            'colleges': colleges_data,
            'total_colleges': len(colleges_data)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch colleges: {str(e)}'}), 500

@app.route('/api/colleges/<college_name>/stats', methods=['GET'])
def get_college_stats(college_name):
    """Get detailed statistics for a college"""
    try:
        # Get review statistics directly from MySQL database using SQLAlchemy ORM
        stats = db.session.query(
            db.func.count(Review.id).label('total_reviews'),
            db.func.avg(Review.rating).label('avg_rating'),
            db.func.count(db.case([(Review.source == 'user_generated', 1)])).label('user_reviews'),
            db.func.count(db.case([(Review.source != 'user_generated', 1)])).label('official_reviews'),
            db.func.count(db.case([(Review.is_anonymous == True, 1)])).label('anonymous_reviews')
        ).filter(
            Review.college_name == college_name,
            Review.is_active == True
        ).first()

        category_stats = db.session.query(
            Review.category,
            db.func.count(Review.id).label('count'),
            db.func.avg(Review.rating).label('avg_rating')
        ).filter(
            Review.college_name == college_name,
            Review.is_active == True
        ).group_by(Review.category).all()

        category_breakdown = {}
        for cat_stat in category_stats:
            category_breakdown[cat_stat.category.value] = {
                'count': cat_stat.count,
                'average_rating': round(cat_stat.avg_rating, 2) if cat_stat.avg_rating else None
            }

        return jsonify({
            'college_name': college_name,
            'statistics': {
                'total_reviews': stats.total_reviews or 0,
                'average_rating': round(stats.avg_rating, 2) if stats.avg_rating else None,
                'user_generated_reviews': stats.user_reviews or 0,
                'official_reviews': stats.official_reviews or 0,
                'anonymous_reviews': stats.anonymous_reviews or 0
            },
            'category_breakdown': category_breakdown,
            'data_sources': {
                'web_scraping_enabled': True,
                'minimum_review_threshold': 50
            }
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to fetch college stats: {str(e)}'}), 500

# User Profile Routes (same as before, with minor updates)
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
        
        # Get user's review statistics
        review_stats = db.session.query(
            db.func.count(Review.id).label('total_reviews'),
            db.func.avg(Review.rating).label('avg_rating'),
            db.func.count(db.case([(Review.is_anonymous == True, 1)])).label('anonymous_reviews')
        ).filter(
            Review.user_id == current_user.user_id,
            Review.is_active == True
        ).first()
        
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
            'social_profiles': social_profiles,
            'review_statistics': {
                'total_reviews': review_stats.total_reviews or 0,
                'average_rating': round(review_stats.avg_rating, 2) if review_stats.avg_rating else None,
                'anonymous_reviews': review_stats.anonymous_reviews or 0
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch profile: {str(e)}'}), 500

# Admin Routes
@app.route('/api/admin/reviews', methods=['GET'])
def admin_get_all_reviews():
    """Admin endpoint to get all reviews with full details"""
    try:
        # In production, add admin authentication
        reviews = Review.query.filter_by(is_active=True).order_by(Review.created_at.desc()).limit(100).all()
        
        reviews_data = []
        for review in reviews:
            user = User.query.filter_by(user_id=review.user_id).first()
            
            review_data = {
                'review_id': review.review_id,
                'content': review.content,
                'rating': review.rating,
                'category': review.category.value,
                'college_name': review.college_name,
                'is_anonymous': review.is_anonymous,
                'source': review.source,
                'source_url': review.source_url,
                'created_at': review.created_at.isoformat(),
                'user_details': {
                    'full_name': user.full_name if user else 'Unknown',
                    'email': user.email if user else 'Unknown',
                    'user_status': user.user_status.value if user else 'Unknown'
                }
            }
            
            reviews_data.append(review_data)
        
        return jsonify({
            'reviews': reviews_data,
            'total_count': len(reviews_data)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Failed to fetch admin reviews: {str(e)}'}), 500

@app.route('/api/admin/scrape-reviews', methods=['POST'])
def admin_trigger_scraping():
    """Admin endpoint to manually trigger review scraping"""
    try:
        data = request.get_json()
        college_name = data.get('college_name')
        
        if not college_name:
            return jsonify({'error': 'college_name is required'}), 400
        
        # Force scraping regardless of current count
        official_reviews = fetch_official_reviews(college_name, limit=50)
        
        # Store reviews
        system_user = User.query.filter_by(email='system@verasuni.com').first()
        if not system_user:
            system_user = User(
                full_name='System',
                email='system@verasuni.com',
                user_status=UserStatus.UNVERIFIED
            )
            db.session.add(system_user)
            db.session.commit()
        
        added_count = 0
        for review_data in official_reviews:
            try:
                review = Review(
                    user_id=system_user.user_id,
                    content=review_data['content'],
                    rating=review_data.get('rating'),
                    category=ReviewCategory(review_data.get('category', 'overall')),
                    college_name=college_name,
                    is_anonymous=True,
                    source=review_data.get('source', 'web_scraped'),
                    source_url=review_data.get('source_url')
                )
                
                db.session.add(review)
                added_count += 1
                
            except Exception as e:
                print(f"Error storing scraped review: {e}")
                continue
        
        db.session.commit()
        
        return jsonify({
            'message': f'Successfully scraped and added {added_count} reviews',
            'college_name': college_name,
            'added_count': added_count
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Scraping failed: {str(e)}'}), 500

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
        print("New features:")
        print("- ✅ Anonymous reviews support")
        print("- ✅ Web scraping fallback for low review count")
        print("- ✅ Multiple review categories")
        print("- ✅ Official data integration")

if __name__ == '__main__':
    init_db()
    print("Starting Verasuni Enhanced Backend Server...")
    print("🔥 NEW FEATURES:")
    print("- 🎭 Anonymous Review Posting")
    print("- 🕷️ Automatic Web Scraping Fallback") 
    print("- 📊 College Statistics & Analytics")
    print("- 🎯 Review Categories (Academics, Campus Life, etc.)")
    print("")
    print("Available endpoints:")
    print("- POST /api/reviews - Post review (with anonymous option)")
    print("- GET /api/reviews/<college_name> - Get college reviews")
    print("- GET /api/user/reviews - Get user's reviews")
    print("- GET /api/colleges - List all colleges")
    print("- GET /api/colleges/<college_name>/stats - College statistics")
    print("- POST /api/admin/scrape-reviews - Manual scraping trigger")
    
    app.run(debug=True, host='0.0.0.0', port=5000)