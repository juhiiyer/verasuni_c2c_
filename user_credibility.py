# Verasuni User Credibility System
# Phase 2 of the Verasuni Algorithm
# Team: Hackstreet Boys ft. Gaurav

from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

class UserType(str, Enum):
    STUDENT = "student"
    ALUMNI = "alumni"
    FACULTY = "faculty"
    PARENT = "parent"
    UNVERIFIED = "unverified"

class VerificationLevel(str, Enum):
    GOLD = "gold"          # 95-100% confidence
    SILVER = "silver"      # 80-94% confidence  
    BRONZE = "bronze"      # 60-79% confidence
    BASIC = "basic"        # 40-59% confidence
    UNVERIFIED = "unverified"  # <40% confidence

@dataclass
class CredibilityFactors:
    email_verification: float = 0.0      # 0.0 - 1.0
    document_verification: float = 0.0   # 0.0 - 1.0
    social_verification: float = 0.0     # 0.0 - 1.0
    behavioral_score: float = 0.0        # 0.0 - 1.0
    contribution_quality: float = 0.0    # 0.0 - 1.0
    peer_validation: float = 0.0         # 0.0 - 1.0

class UserCredibilityEngine:
    """
    Advanced Multi-Factor User Credibility System for Verasuni
    
    Features:
    - 6-dimensional credibility assessment
    - Dynamic user type multipliers
    - Document quality verification
    - Social profile cross-verification
    - Behavioral pattern analysis
    - Community engagement scoring
    - Personalized improvement recommendations
    """
    
    def __init__(self):
        # Email domain trust levels
        self.trusted_domains = {
            'student': {
                'high_trust': ['.edu', '.ac.in', 'vitstudent.ac.in', 'iit.ac.in', 'nit.ac.in'],
                'medium_trust': ['student.', '.edu.in', 'college.'],
                'low_trust': ['gmail.com', 'yahoo.com', 'hotmail.com']
            },
            'alumni': {
                'high_trust': ['.edu', '.ac.in', 'vitstudent.ac.in', 'alumni.'],
                'medium_trust': ['gmail.com', 'yahoo.com', 'outlook.com'],
                'low_trust': ['temp', '10minute', 'throwaway']
            }
        }
        
        # User type credibility multipliers
        self.base_multipliers = {
            UserType.ALUMNI: 1.2,      # Alumni get 20% credibility bonus
            UserType.STUDENT: 1.0,     # Students baseline credibility
            UserType.FACULTY: 1.5,     # Faculty get 50% credibility bonus
            UserType.PARENT: 0.8,      # Parents 20% lower baseline
            UserType.UNVERIFIED: 0.3   # Unverified users much lower
        }
        
        # Document verification weights
        self.document_weights = {
            'student_id': 0.4,
            'enrollment_certificate': 0.3,
            'degree_certificate': 0.5,
            'alumni_id': 0.4,
            'employee_id': 0.5,
            'linkedin_profile': 0.3
        }

    def calculate_user_credibility(
        self, 
        user_data: Dict[str, Any],
        contribution_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive user credibility score
        
        Args:
            user_data: User profile information
            contribution_history: User's past contributions and reviews
            
        Returns:
            Comprehensive credibility assessment with scores and recommendations
        """
        
        # Extract user information
        user_type = UserType(user_data.get('user_type', 'unverified'))
        email = user_data.get('email', '')
        documents = user_data.get('documents', {})
        social_profiles = user_data.get('social_profiles', {})
        
        # Initialize credibility factors
        factors = CredibilityFactors()
        
        # Calculate individual credibility factors
        factors.email_verification = self._verify_email_credibility(email, user_type)
        factors.document_verification = self._verify_documents_credibility(documents, user_type)
        factors.social_verification = self._verify_social_credibility(social_profiles, user_data)
        factors.behavioral_score = self._calculate_behavioral_score(user_data, contribution_history)
        factors.contribution_quality = self._assess_contribution_quality(contribution_history)
        factors.peer_validation = self._calculate_peer_validation(user_data, contribution_history)
        
        # Calculate weighted credibility score (weights based on Verasuni research)
        weighted_score = (
            factors.email_verification * 0.25 +      # 25% weight
            factors.document_verification * 0.30 +   # 30% weight (most important)
            factors.social_verification * 0.15 +     # 15% weight
            factors.behavioral_score * 0.15 +        # 15% weight
            factors.contribution_quality * 0.10 +    # 10% weight
            factors.peer_validation * 0.05           # 5% weight
        )
        
        # Apply user type multiplier
        base_multiplier = self.base_multipliers.get(user_type, 0.5)
        final_score = weighted_score * base_multiplier
        
        # Determine verification level
        verification_level = self._determine_verification_level(final_score)
        
        # Calculate confidence percentage
        confidence = min(100, int(final_score * 100))
        
        # Generate improvement recommendations
        recommendations = self._generate_credibility_recommendations(factors, verification_level)
        
        return {
            'user_id': user_data.get('user_id'),
            'credibility_score': round(final_score, 3),
            'confidence_percentage': confidence,
            'verification_level': verification_level.value,
            'user_type': user_type.value,
            'factors': {
                'email_verification': round(factors.email_verification, 3),
                'document_verification': round(factors.document_verification, 3),
                'social_verification': round(factors.social_verification, 3),
                'behavioral_score': round(factors.behavioral_score, 3),
                'contribution_quality': round(factors.contribution_quality, 3),
                'peer_validation': round(factors.peer_validation, 3)
            },
            'base_multiplier': base_multiplier,
            'weighted_score': round(weighted_score, 3),
            'calculated_at': datetime.now().isoformat(),
            'recommendations': recommendations
        }
    
    def _verify_email_credibility(self, email: str, user_type: UserType) -> float:
        """Verify email domain credibility based on user type"""
        if not email or '@' not in email:
            return 0.0
        
        domain = email.split('@')[1].lower()
        trust_levels = self.trusted_domains.get(user_type.value, {})
        
        # Check trust levels
        if any(trusted in domain for trusted in trust_levels.get('high_trust', [])):
            return 0.9  # High trust institutional domains
        elif any(trusted in domain for trusted in trust_levels.get('medium_trust', [])):
            return 0.6  # Medium trust domains
        elif any(untrusted in domain for untrusted in trust_levels.get('low_trust', [])):
            return 0.3  # Low trust domains
        else:
            return 0.4  # Unknown domain gets neutral score
    
    def _verify_documents_credibility(self, documents: Dict[str, Any], user_type: UserType) -> float:
        """Verify submitted documents credibility"""
        if not documents:
            return 0.0
        
        total_weight = 0.0
        achieved_weight = 0.0
        
        # Required documents by user type
        required_docs = {
            UserType.STUDENT: ['student_id', 'enrollment_certificate'],
            UserType.ALUMNI: ['degree_certificate', 'alumni_id'],
            UserType.FACULTY: ['employee_id', 'transcript'],
            UserType.PARENT: ['linkedin_profile']
        }
        
        required = required_docs.get(user_type, [])
        
        for doc_type in required:
            weight = self.document_weights.get(doc_type, 0.2)
            total_weight += weight
            
            if doc_type in documents:
                doc_quality = self._assess_document_quality(documents[doc_type])
                achieved_weight += weight * doc_quality
        
        # Bonus for additional documents
        bonus_docs = [doc for doc in documents.keys() if doc not in required]
        for doc_type in bonus_docs:
            weight = self.document_weights.get(doc_type, 0.1)
            doc_quality = self._assess_document_quality(documents[doc_type])
            achieved_weight += weight * doc_quality * 0.5  # Bonus docs worth 50%
        
        return min(1.0, achieved_weight / max(total_weight, 0.1))
    
    def _assess_document_quality(self, document: Dict[str, Any]) -> float:
        """Assess quality of individual document"""
        quality_factors = []
        
        # File format check - legitimate document formats
        file_url = document.get('url', '')
        if file_url.endswith(('.pdf', '.jpg', '.png', '.jpeg')):
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.4)
        
        # Upload recency - recent uploads are more trustworthy
        if document.get('uploaded_at'):
            quality_factors.append(0.9)
        else:
            quality_factors.append(0.5)
        
        # File size check - reasonable size indicates legitimate document
        file_size = document.get('file_size', 0)
        if 50000 <= file_size <= 5000000:  # 50KB to 5MB reasonable range
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.4)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
    
    def _verify_social_credibility(self, social_profiles: Dict[str, Any], user_data: Dict[str, Any]) -> float:
        """Verify social media profile credibility"""
        if not social_profiles:
            return 0.0
        
        credibility_score = 0.0
        profile_count = 0
        
        for platform, profile in social_profiles.items():
            profile_score = self._assess_social_profile(platform, profile, user_data)
            credibility_score += profile_score
            profile_count += 1
        
        return credibility_score / max(profile_count, 1)
    
    def _assess_social_profile(self, platform: str, profile: Dict[str, Any], user_data: Dict[str, Any]) -> float:
        """Assess individual social media profile credibility"""
        # Platform credibility weights
        platform_weights = {
            'linkedin': 0.9,    # Highest credibility for professional profiles
            'facebook': 0.6,    # Medium credibility
            'twitter': 0.5,     # Lower credibility
            'instagram': 0.4    # Lowest credibility for academic verification
        }
        
        base_weight = platform_weights.get(platform.lower(), 0.3)
        
        # Profile completeness factors
        completeness_score = 0.0
        if profile.get('profile_picture'):
            completeness_score += 0.2
        if profile.get('bio') and len(profile['bio']) > 20:
            completeness_score += 0.3
        if profile.get('followers', 0) > 10:
            completeness_score += 0.2
        if profile.get('posts', 0) > 5:
            completeness_score += 0.3
        
        # Name consistency check
        user_name = user_data.get('full_name', '').lower()
        profile_name = profile.get('name', '').lower()
        name_similarity = self._calculate_name_similarity(user_name, profile_name)
        
        # Combined profile score
        final_score = base_weight * (completeness_score + name_similarity * 0.3)
        return min(1.0, final_score)
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names"""
        if not name1 or not name2:
            return 0.0
        
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_behavioral_score(self, user_data: Dict[str, Any], contribution_history: Optional[List[Dict[str, Any]]]) -> float:
        """Calculate behavioral credibility based on user patterns"""
        if not contribution_history:
            return 0.5  # Neutral score for new users
        
        behavioral_factors = []
        
        # Account age factor
        account_age_score = self._assess_account_age(user_data)
        behavioral_factors.append(account_age_score)
        
        # Contribution consistency
        consistency_score = self._assess_contribution_consistency(contribution_history)
        behavioral_factors.append(consistency_score)
        
        # Response to feedback
        feedback_score = self._assess_feedback_responsiveness(contribution_history)
        behavioral_factors.append(feedback_score)
        
        return sum(behavioral_factors) / len(behavioral_factors)
    
    def _assess_account_age(self, user_data: Dict[str, Any]) -> float:
        """Assess credibility based on account age"""
        created_at = user_data.get('created_at')
        if not created_at:
            return 0.3
        
        account_age = datetime.now() - datetime.fromisoformat(created_at)
        age_days = account_age.days
        
        # Longer accounts generally more credible
        if age_days > 365:      # Over 1 year
            return 0.9
        elif age_days > 180:    # Over 6 months
            return 0.7
        elif age_days > 30:     # Over 1 month
            return 0.6
        else:                   # New accounts
            return 0.4
    
    def _assess_contribution_consistency(self, contributions: List[Dict[str, Any]]) -> float:
        """Assess consistency of user contributions"""
        if len(contributions) < 2:
            return 0.5
        
        # Simple consistency based on number of contributions
        if len(contributions) >= 5:
            return 0.8      # Very consistent
        elif len(contributions) >= 3:
            return 0.7      # Consistent
        else:
            return 0.6      # Somewhat consistent
    
    def _assess_feedback_responsiveness(self, contributions: List[Dict[str, Any]]) -> float:
        """Assess how user responds to community feedback"""
        total_feedback = 0
        positive_responses = 0
        
        for contribution in contributions:
            feedback = contribution.get('community_feedback', {})
            upvotes = feedback.get('upvotes', 0)
            downvotes = feedback.get('downvotes', 0)
            
            if upvotes + downvotes > 0:
                total_feedback += 1
                if upvotes > downvotes:
                    positive_responses += 1
        
        if total_feedback == 0:
            return 0.5  # Neutral for no feedback
        
        return positive_responses / total_feedback
    
    def _assess_contribution_quality(self, contribution_history: Optional[List[Dict[str, Any]]]) -> float:
        """Assess quality of user's contributions"""
        if not contribution_history:
            return 0.5
        
        quality_scores = []
        
        for contribution in contribution_history:
            # Content length and detail quality
            content = contribution.get('content', '')
            length_score = min(1.0, len(content) / 200)  # Good if >200 chars
            
            # Community engagement
            engagement = contribution.get('community_feedback', {})
            upvotes = engagement.get('upvotes', 0)
            total_votes = upvotes + engagement.get('downvotes', 0)
            
            engagement_score = (upvotes / max(total_votes, 1)) if total_votes > 0 else 0.5
            
            # Helpfulness ratings
            helpful_score = contribution.get('helpfulness_rating', 2.5) / 5.0
            
            # Combined quality score
            contrib_quality = (length_score * 0.3 + engagement_score * 0.4 + helpful_score * 0.3)
            quality_scores.append(contrib_quality)
        
        return sum(quality_scores) / len(quality_scores)
    
    def _calculate_peer_validation(self, user_data: Dict[str, Any], contribution_history: Optional[List[Dict[str, Any]]]) -> float:
        """Calculate peer validation score"""
        if not contribution_history:
            return 0.5
        
        # Count peer endorsements
        total_endorsements = sum(c.get('peer_endorsements', 0) for c in contribution_history)
        total_contributions = len(contribution_history)
        
        endorsement_ratio = total_endorsements / max(total_contributions, 1)
        
        # Referral bonus
        referrals = user_data.get('referrals_made', 0)
        referral_bonus = min(0.3, referrals * 0.05)  # Max 0.3 bonus
        
        return min(1.0, endorsement_ratio + referral_bonus)
    
    def _determine_verification_level(self, credibility_score: float) -> VerificationLevel:
        """Determine verification level based on credibility score"""
        if credibility_score >= 0.95:
            return VerificationLevel.GOLD
        elif credibility_score >= 0.80:
            return VerificationLevel.SILVER  
        elif credibility_score >= 0.60:
            return VerificationLevel.BRONZE
        elif credibility_score >= 0.40:
            return VerificationLevel.BASIC
        else:
            return VerificationLevel.UNVERIFIED
    
    def _generate_credibility_recommendations(self, factors: CredibilityFactors, level: VerificationLevel) -> List[str]:
        """Generate personalized recommendations to improve credibility"""
        recommendations = []
        
        if factors.email_verification < 0.6:
            recommendations.append("Verify your email with an institutional domain (.edu, .ac.in)")
        
        if factors.document_verification < 0.5:
            recommendations.append("Upload valid documents (student ID, degree certificate, etc.)")
        
        if factors.social_verification < 0.4:
            recommendations.append("Link your LinkedIn or other professional profiles")
        
        if factors.contribution_quality < 0.5:
            recommendations.append("Write more detailed, helpful reviews and responses")
        
        if factors.peer_validation < 0.4:
            recommendations.append("Engage more with the community and help verify others' information")
        
        if level == VerificationLevel.UNVERIFIED:
            recommendations.append("Complete basic verification steps to unlock more platform features")
        
        return recommendations

# Demo and Testing Functions
def create_sample_data():
    """Create sample user data for testing"""
    
    sample_users = [
        {
            'user_id': 'user_001',
            'user_type': 'student',
            'email': 'john.doe@vit.ac.in',
            'full_name': 'John Doe',
            'created_at': '2023-08-15T10:30:00',
            'documents': {
                'student_id': {
                    'url': 'https://example.com/student_id.pdf',
                    'uploaded_at': '2023-08-16T14:20:00',
                    'file_size': 1024000
                },
                'enrollment_certificate': {
                    'url': 'https://example.com/enrollment.pdf',
                    'uploaded_at': '2023-08-16T14:25:00',
                    'file_size': 856000
                }
            },
            'social_profiles': {
                'linkedin': {
                    'name': 'John Doe',
                    'bio': 'Computer Science student at VIT Vellore',
                    'followers': 150,
                    'posts': 20,
                    'profile_picture': True
                }
            },
            'referrals_made': 3
        },
        {
            'user_id': 'user_002',
            'user_type': 'alumni',
            'email': 'jane.smith@gmail.com',
            'full_name': 'Jane Smith',
            'created_at': '2022-05-10T09:15:00',
            'documents': {
                'degree_certificate': {
                    'url': 'https://example.com/degree.pdf',
                    'uploaded_at': '2022-05-12T16:45:00',
                    'file_size': 2048000
                },
                'alumni_id': {
                    'url': 'https://example.com/alumni_id.jpg',
                    'uploaded_at': '2022-05-12T16:50:00',
                    'file_size': 512000
                }
            },
            'social_profiles': {
                'linkedin': {
                    'name': 'Jane Smith',
                    'bio': 'Software Engineer at Google | VIT Alumnus',
                    'followers': 500,
                    'posts': 45,
                    'profile_picture': True
                }
            },
            'referrals_made': 8
        }
    ]
    
    contribution_histories = [
        [  # User 1 contributions
            {
                'content': 'VIT has excellent placement opportunities. The computer science department is particularly strong with good faculty and modern labs. I would definitely recommend this college for CS students.',
                'created_at': '2024-01-15T12:00:00',
                'community_feedback': {'upvotes': 25, 'downvotes': 2},
                'helpfulness_rating': 4.2,
                'peer_endorsements': 3
            },
            {
                'content': 'The hostel facilities are decent but could be improved. Food quality varies by mess but is generally acceptable for college standards.',
                'created_at': '2024-02-10T14:30:00', 
                'community_feedback': {'upvotes': 18, 'downvotes': 5},
                'helpfulness_rating': 3.8,
                'peer_endorsements': 1
            }
        ],
        [  # User 2 contributions
            {
                'content': 'As an alumnus who graduated 3 years ago, I can confidently say VIT prepared me well for the industry. The curriculum is updated regularly and faculty are supportive. Got placed in Google through campus recruitment.',
                'created_at': '2023-09-20T10:15:00',
                'community_feedback': {'upvotes': 42, 'downvotes': 1},
                'helpfulness_rating': 4.7,
                'peer_endorsements': 8
            },
            {
                'content': 'Placement support is really strong at VIT. The placement cell is very active and helps with interview preparation. I got offers from multiple companies including Google, Microsoft, and Amazon.',
                'created_at': '2023-11-05T16:20:00',
                'community_feedback': {'upvotes': 35, 'downvotes': 0},
                'helpfulness_rating': 4.9,
                'peer_endorsements': 5
            }
        ]
    ]
    
    return sample_users, contribution_histories

def demo_credibility_system():
    """Run a comprehensive demo of the credibility system"""
    
    engine = UserCredibilityEngine()
    sample_users, contribution_histories = create_sample_data()
    
    print("🔍 VERASUNI USER CREDIBILITY SYSTEM DEMO")
    print("=" * 60)
    print("Team: Hackstreet Boys ft. Gaurav")
    print("Algorithm: Phase 2 - Multi-Factor User Verification")
    print("=" * 60)

    for i, user in enumerate(sample_users):
        print(f"\n📊 USER {i+1} - {user['user_type'].upper()} CREDIBILITY ANALYSIS")
        print("-" * 50)
        
        result = engine.calculate_user_credibility(user, contribution_histories[i])
        
        # Display main results
        print(f"👤 User ID: {result['user_id']}")
        print(f"🎯 Credibility Score: {result['credibility_score']} ({result['confidence_percentage']}% confidence)")
        print(f"🏅 Verification Level: {result['verification_level'].upper()}")
        print(f"👨‍🎓 User Type: {result['user_type'].upper()} (Multiplier: {result['base_multiplier']}x)")
        print(f"⚖️ Weighted Base Score: {result['weighted_score']}")
        
        print(f"\n📈 CREDIBILITY FACTOR BREAKDOWN:")
        factors = result['factors']
        
        # Create visual progress bars
        for factor, score in factors.items():
            factor_name = factor.replace('_', ' ').title()
            bar_length = int(score * 20)  # Scale to 20 characters
            bar = "█" * bar_length + "░" * (20 - bar_length)
            percentage = int(score * 100)
            print(f"  {factor_name:20} │{bar}│ {score:.3f} ({percentage}%)")
        
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS ({len(result['recommendations'])}):")
        if result['recommendations']:
            for j, rec in enumerate(result['recommendations'], 1):
                print(f"  {j}. {rec}")
        else:
            print("  ✅ No recommendations - excellent credibility profile!")
        
        print(f"\n🎖️ VERIFICATION BADGES EARNED:")
        
        # Show badges based on factor scores
        badges = []
        if factors['email_verification'] >= 0.8:
            badges.append("📧 Verified Email")
        if factors['document_verification'] >= 0.7:
            badges.append("📄 Document Verified")
        if factors['social_verification'] >= 0.6:
            badges.append("🔗 Social Verified")
        if factors['contribution_quality'] >= 0.7:
            badges.append("⭐ Quality Contributor")
        if factors['peer_validation'] >= 0.6:
            badges.append("🤝 Community Trusted")
        if result['credibility_score'] >= 0.8:
            badges.append("🏆 High Credibility")
        
        if badges:
            for badge in badges:
                print(f"  {badge}")
        else:
            print("  🥉 Basic Verification - Complete profile to earn more badges!")

    print("\n" + "=" * 60)
    print("✅ VERASUNI CREDIBILITY SYSTEM ANALYSIS COMPLETE!")

    print("\n📊 ALGORITHM PERFORMANCE SUMMARY:")
    for i, user in enumerate(sample_users):
        result = engine.calculate_user_credibility(user, contribution_histories[i])
        print(f"  📈 User {i+1} ({user['user_type']}): {result['confidence_percentage']}% confidence → {result['verification_level'].upper()}")

    print("\n🎯 READY FOR HACKATHON INTEGRATION!")
    print("🔗 Integration: Import UserCredibilityEngine into main Verasuni algorithm")

if __name__ == "__main__":
    demo_credibility_system()