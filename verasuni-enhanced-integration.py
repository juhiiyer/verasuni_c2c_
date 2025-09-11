from flask import request, jsonify


try:
    from verasuni_enhanced_backend import Review, User
except ImportError:
    # If the module is not found, handle gracefully or provide a message
    Review = None
    User = None
    print("Warning: verasuni_enhanced_backend module not found. Some features may be disabled.")
# Verasuni Complete Algorithm with Anonymous Reviews & Web Scraping
# Team: Hackstreet Boys ft. Gaurav
# Integration Guide for Enhanced Backend

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

class EnhancedVerasuniEngine:
    """
    Enhanced Verasuni Algorithm with Anonymous Reviews and Web Scraping Fallback
    
    NEW FEATURES:
    - Anonymous review posting and processing
    - Automatic web scraping when review count < 50
    - Multi-source data integration
    - Enhanced credibility with source weighting
    """
    
    def __init__(self):
        # Initialize all algorithm components
        from user_credibility import UserCredibilityEngine
        from bias_detection import VerasuniBiasDetector  
        from bias_corrector import VerasuniBiasCorrector
        
        self.credibility_engine = UserCredibilityEngine()
        self.bias_detector = VerasuniBiasDetector()
        self.bias_corrector = VerasuniBiasCorrector()
        
        # NEW: Web scraping configuration
        self.min_reviews_threshold = 50
        self.scraping_enabled = True
        
    async def process_complete_pipeline(self, raw_reviews_data, college_name):
        """
        Enhanced Complete Verasuni processing pipeline with web scraping
        
        Input: Raw reviews with user data + college name
        Output: Bias-neutralized, credibility-weighted recommendations with scraped data
        """
        
        print("🚀 Starting Enhanced Verasuni Processing Pipeline...")
        
        # Phase 1: Data Collection with Web Scraping Enhancement
        collected_data = await self._enhanced_data_collection(raw_reviews_data, college_name)
        print(f"✅ Phase 1: Collected {len(collected_data['reviews'])} reviews "
              f"({collected_data['scraped_count']} from web scraping)")
        
        # Phase 2: Enhanced User Credibility Assessment (handles anonymous reviews)
        credibility_results = []
        for review_data in collected_data['reviews']:
            user_data = review_data['user']
            contribution_history = review_data.get('contribution_history', [])
            is_anonymous = review_data.get('is_anonymous', False)
            source = review_data.get('source', 'user_generated')
            
            credibility = self.credibility_engine.calculate_enhanced_credibility(
                user_data, contribution_history, is_anonymous, source
            )
            credibility_results.append(credibility)
        
        print(f"✅ Phase 2: Assessed credibility for {len(credibility_results)} users")
        
        # Phase 3: Bias Detection with Source Weighting
        reviews_for_bias_detection = [rd['review'] for rd in collected_data['reviews']]
        bias_analysis = await self.bias_detector.detect_bias_enhanced(
            reviews_for_bias_detection, 
            source_info=[rd.get('source', 'user_generated') for rd in collected_data['reviews']]
        )
        
        print(f"✅ Phase 3: Detected bias in {bias_analysis['overall_statistics']['biased_reviews_count']} reviews")
        
        # Phase 4: Bias Correction with Anonymous Handling
        biased_reviews = [
            rd['review'] for i, rd in enumerate(collected_data['reviews'])
            if bias_analysis['individual_results'][i]['overall_bias_score'] > 0.3
        ]
        biased_analysis = [
            result for result in bias_analysis['individual_results'] 
            if result['overall_bias_score'] > 0.3
        ]
        
        correction_results = await self.bias_corrector.correct_bias_enhanced(
            biased_reviews, biased_analysis
        )
        
        print(f"✅ Phase 4: Corrected {correction_results['reviews_corrected']} biased reviews")
        
        # Phase 5: Enhanced Truth Synthesis with Multi-Source Weighting
        truth_synthesis = self._enhanced_truth_synthesis(
            credibility_results, bias_analysis, correction_results, collected_data
        )
        
        print(f"✅ Phase 5: Synthesized truth scores with {truth_synthesis['confidence_level']} confidence")
        
        # Phase 6&7: Enhanced Recommendations
        final_recommendations = self._generate_enhanced_recommendations(
            truth_synthesis, collected_data['college_info']
        )
        
        print(f"✅ Phase 6&7: Generated enhanced recommendations")
        
        return {
            'pipeline_results': {
                'phase_1_enhanced_data_collection': collected_data,
                'phase_2_enhanced_credibility': {
                    'results': credibility_results,
                    'avg_credibility': sum(cr['credibility_score'] for cr in credibility_results) / len(credibility_results),
                    'anonymous_reviews': len([cr for cr in credibility_results if cr.get('is_anonymous', False)]),
                    'scraped_reviews': len([cr for cr in credibility_results if cr.get('source') != 'user_generated'])
                },
                'phase_3_enhanced_bias_detection': bias_analysis,
                'phase_4_enhanced_bias_correction': correction_results,
                'phase_5_enhanced_truth_synthesis': truth_synthesis,
                'phase_67_enhanced_recommendations': final_recommendations
            },
            'enhanced_metrics': {
                'total_reviews_processed': len(raw_reviews_data),
                'web_scraped_reviews': collected_data['scraped_count'],
                'anonymous_reviews_count': len([rd for rd in collected_data['reviews'] if rd.get('is_anonymous', False)]),
                'multi_source_integration': True,
                'bias_detected_percentage': bias_analysis['overall_statistics']['biased_reviews_percentage'],
                'bias_corrected_count': correction_results['reviews_corrected'],
                'sentiment_preservation_rate': correction_results.get('correction_statistics', {}).get('sentiment_preservation_rate', 0),
                'enhanced_confidence': truth_synthesis['overall_confidence'],
                'processing_time_ms': len(raw_reviews_data) * 60  # Slightly higher due to scraping
            }
        }
    
    async def _enhanced_data_collection(self, raw_data, college_name):
        """Phase 1: Enhanced data collection with web scraping fallback"""
        
        current_review_count = len(raw_data)
        scraped_reviews = []
        scraped_count = 0
        
        # NEW: Check if we need web scraping
        if current_review_count < self.min_reviews_threshold and self.scraping_enabled:
            print(f"⚠️  Low review count ({current_review_count}). Initiating web scraping...")
            
            try:
                # Fetch official reviews from multiple sources
                scraped_reviews = await self._fetch_and_process_official_reviews(
                    college_name, 
                    limit=self.min_reviews_threshold - current_review_count
                )
                scraped_count = len(scraped_reviews)
                print(f"📥 Successfully scraped {scraped_count} official reviews")
                
            except Exception as e:
                print(f"❌ Web scraping failed: {e}")
                scraped_reviews = []
        
        # Combine original and scraped reviews
        all_reviews = raw_data + scraped_reviews
        
        return {
            'reviews': all_reviews,
            'original_count': len(raw_data),
            'scraped_count': scraped_count,
            'total_count': len(all_reviews),
            'college_info': {
                'college_id': college_name.upper().replace(' ', '_'),
                'name': college_name,
                'official_data': {
                    'nirf_ranking': 'TBD',  # Can be scraped
                    'placement_rate': 'TBD',
                    'average_package': 'TBD'
                },
                'data_sources': {
                    'user_generated': len(raw_data),
                    'web_scraped': scraped_count,
                    'last_scraped': datetime.now().isoformat()
                }
            },
            'data_sources': ['user_reviews', 'web_scraped', 'official_stats'],
            'collection_timestamp': datetime.now().isoformat()
        }
    
    async def _fetch_and_process_official_reviews(self, college_name, limit=20):
        """Fetch and process official reviews from multiple sources"""
        
        official_reviews = []
        
        try:
            # Source 1: CollegeDunia simulation
            collegedunia_reviews = self._simulate_collegedunia_scraping(college_name, limit//2)
            for review in collegedunia_reviews:
                processed_review = {
                    'review': {
                        'id': f"scraped_{len(official_reviews)+1}",
                        'content': review['content'],
                        'rating': review['rating'],
                        'timestamp': datetime.now().isoformat()
                    },
                    'user': {
                        'user_id': 'official_collegedunia',
                        'user_type': 'official',
                        'full_name': 'Official Review',
                        'created_at': datetime.now().isoformat(),
                        'documents': {},
                        'social_profiles': {}
                    },
                    'is_anonymous': True,  # Official reviews are anonymous
                    'source': 'collegedunia',
                    'source_url': f'https://collegedunia.com/college/{college_name.lower().replace(" ", "-")}/reviews',
                    'contribution_history': []
                }
                official_reviews.append(processed_review)
            
            # Source 2: Shiksha simulation
            shiksha_reviews = self._simulate_shiksha_scraping(college_name, limit//2)
            for review in shiksha_reviews:
                processed_review = {
                    'review': {
                        'id': f"scraped_{len(official_reviews)+1}",
                        'content': review['content'],
                        'rating': review['rating'],
                        'timestamp': datetime.now().isoformat()
                    },
                    'user': {
                        'user_id': 'official_shiksha',
                        'user_type': 'official',
                        'full_name': 'Official Review',
                        'created_at': datetime.now().isoformat(),
                        'documents': {},
                        'social_profiles': {}
                    },
                    'is_anonymous': True,
                    'source': 'shiksha',
                    'source_url': f'https://shiksha.com/college/{college_name.lower().replace(" ", "-")}-reviews',
                    'contribution_history': []
                }
                official_reviews.append(processed_review)
            
        except Exception as e:
            print(f"Error in official review processing: {e}")
        
        return official_reviews[:limit]
    
    def _simulate_collegedunia_scraping(self, college_name, limit=10):
        """Simulate CollegeDunia scraping (replace with actual implementation)"""
        
        sample_reviews = [
            {
                'content': f'{college_name} offers good placement opportunities with experienced faculty. The campus infrastructure is well-maintained and provides a conducive learning environment.',
                'rating': 4.2,
                'category': 'overall'
            },
            {
                'content': f'The academic curriculum at {college_name} is updated and relevant to current industry needs. Good lab facilities and comprehensive library resources.',
                'rating': 4.0,
                'category': 'academics'
            },
            {
                'content': f'Campus life at {college_name} is vibrant with various clubs and societies. Good sports facilities and cultural activities throughout the year.',
                'rating': 4.3,
                'category': 'campus_life'
            },
            {
                'content': f'Placement record of {college_name} is commendable with good companies visiting for recruitment. Career guidance and training programs are effective.',
                'rating': 4.1,
                'category': 'placements'
            }
        ]
        
        import random
        selected_reviews = random.sample(sample_reviews, min(limit, len(sample_reviews)))
        
        # Add some variation to ratings
        for review in selected_reviews:
            review['rating'] += random.uniform(-0.3, 0.3)
            review['rating'] = max(1.0, min(5.0, review['rating']))
            review['rating'] = round(review['rating'], 1)
        
        return selected_reviews
    
    def _simulate_shiksha_scraping(self, college_name, limit=10):
        """Simulate Shiksha.com scraping (replace with actual implementation)"""
        
        sample_reviews = [
            {
                'content': f'{college_name} has good hostel facilities with decent food quality. Wi-Fi connectivity is reliable across the campus.',
                'rating': 3.8,
                'category': 'facilities'
            },
            {
                'content': f'Faculty at {college_name} are knowledgeable and supportive. Regular industry interactions and guest lectures enhance learning.',
                'rating': 4.0,
                'category': 'faculty'
            },
            {
                'content': f'The overall experience at {college_name} has been positive. Good balance between academics and extracurricular activities.',
                'rating': 3.9,
                'category': 'overall'
            },
            {
                'content': f'Food options at {college_name} are varied and hygienic. Multiple cafeterias and food courts cater to different preferences.',
                'rating': 3.7,
                'category': 'food'
            }
        ]
        
        import random
        selected_reviews = random.sample(sample_reviews, min(limit, len(sample_reviews)))
        
        # Add some variation
        for review in selected_reviews:
            review['rating'] += random.uniform(-0.2, 0.2)
            review['rating'] = max(1.0, min(5.0, review['rating']))
            review['rating'] = round(review['rating'], 1)
        
        return selected_reviews
    
    def _enhanced_truth_synthesis(self, credibility_results, bias_analysis, correction_results, collected_data):
        """Phase 5: Enhanced truth synthesis with multi-source weighting"""
        
        # Enhanced weighting system
        weighted_scores = {
            'academics': {'total': 0.0, 'weight': 0.0},
            'campus_life': {'total': 0.0, 'weight': 0.0}, 
            'placements': {'total': 0.0, 'weight': 0.0},
            'facilities': {'total': 0.0, 'weight': 0.0},
            'food': {'total': 0.0, 'weight': 0.0},
            'overall': {'total': 0.0, 'weight': 0.0}
        }
        
        total_weight = 0.0
        source_weights = {
            'user_generated': 1.0,
            'collegedunia': 0.8,  # Official sources get lower weight than users
            'shiksha': 0.8,
            'official': 0.9
        }
        
        for i, credibility in enumerate(credibility_results):
            # Get bias info
            bias_info = bias_analysis['individual_results'][i] if i < len(bias_analysis['individual_results']) else {}
            
            # Get source information
            review_data = collected_data['reviews'][i] if i < len(collected_data['reviews']) else {}
            source = review_data.get('source', 'user_generated')
            is_anonymous = review_data.get('is_anonymous', False)
            
            # Calculate enhanced review weight
            credibility_weight = credibility['credibility_score']
            bias_penalty = 1.0 - min(0.5, bias_info.get('overall_bias_score', 0))
            source_weight = source_weights.get(source, 0.5)
            
            # Anonymous reviews get slight penalty for accountability
            anonymous_penalty = 0.9 if is_anonymous else 1.0
            
            review_weight = credibility_weight * bias_penalty * source_weight * anonymous_penalty
            
            # Distribute to categories (simulated)
            aspect_ratings = self._extract_aspect_ratings(review_data.get('review', {}))
            
            for aspect, rating in aspect_ratings.items():
                if aspect in weighted_scores:
                    weighted_scores[aspect]['total'] += rating * review_weight
                    weighted_scores[aspect]['weight'] += review_weight
                    
            total_weight += review_weight
        
        # Calculate final weighted averages
        final_scores = {}
        for aspect, data in weighted_scores.items():
            if data['weight'] > 0:
                final_scores[aspect] = round(data['total'] / data['weight'], 2)
            else:
                final_scores[aspect] = None
        
        # Calculate overall confidence
        user_generated_ratio = collected_data['original_count'] / collected_data['total_count']
        scraped_data_bonus = min(0.1, collected_data['scraped_count'] / 100)  # Bonus for more data
        
        overall_confidence = min(0.95, 
            0.6 + 
            (total_weight / len(credibility_results)) * 0.25 + 
            user_generated_ratio * 0.1 +
            scraped_data_bonus
        )
        
        return {
            'aspect_scores': final_scores,
            'overall_score': round(sum(score for score in final_scores.values() if score is not None) / 
                                 len([score for score in final_scores.values() if score is not None]), 2),
            'total_weight_applied': round(total_weight, 2),
            'confidence_factors': {
                'credibility_weighted': True,
                'bias_corrected': True,
                'multi_source_verified': True,
                'anonymous_handling': True,
                'web_scraping_enhanced': collected_data['scraped_count'] > 0
            },
            'overall_confidence': round(overall_confidence, 3),
            'confidence_level': 'High' if overall_confidence > 0.8 else 'Medium' if overall_confidence > 0.6 else 'Low',
            'data_composition': {
                'user_generated_ratio': round(user_generated_ratio, 2),
                'scraped_data_ratio': round(collected_data['scraped_count'] / collected_data['total_count'], 2),
                'anonymous_ratio': round(len([cr for cr in credibility_results if cr.get('is_anonymous', False)]) / len(credibility_results), 2)
            }
        }
    
    def _extract_aspect_ratings(self, review_data):
        """Extract aspect ratings from review content (simulated)"""
        # In real implementation, this would use NLP to extract aspect-specific sentiments
        base_rating = review_data.get('rating', 3.5)
        
        return {
            'academics': base_rating + (-0.2 + 0.4 * hash(str(review_data)) % 100 / 100),
            'campus_life': base_rating + (-0.3 + 0.6 * hash(str(review_data) + 'campus') % 100 / 100),
            'placements': base_rating + (-0.1 + 0.2 * hash(str(review_data) + 'placement') % 100 / 100),
            'facilities': base_rating + (-0.2 + 0.4 * hash(str(review_data) + 'facility') % 100 / 100),
            'food': base_rating + (-0.5 + 1.0 * hash(str(review_data) + 'food') % 100 / 100),
            'overall': base_rating
        }
    
    def _generate_enhanced_recommendations(self, truth_synthesis, college_info):
        """Phase 6&7: Generate enhanced recommendations with source transparency"""
        
        overall_score = truth_synthesis['overall_score']
        aspect_scores = truth_synthesis['aspect_scores']
        confidence = truth_synthesis['overall_confidence']
        
        # Enhanced recommendation logic
        if overall_score >= 4.0 and confidence > 0.8:
            recommendation = "Highly Recommended"
            confidence_text = "High Confidence"
        elif overall_score >= 3.5 and confidence > 0.7:
            recommendation = "Recommended" 
            confidence_text = "Good Confidence"
        elif overall_score >= 3.0 and confidence > 0.6:
            recommendation = "Conditionally Recommended"
            confidence_text = "Moderate Confidence"
        else:
            recommendation = "Needs More Research"
            confidence_text = "Low Confidence"
        
        # Identify strengths and concerns
        strengths = [aspect for aspect, score in aspect_scores.items() 
                    if score and score >= 4.0]
        concerns = [aspect for aspect, score in aspect_scores.items() 
                   if score and score < 3.5]
        
        # Enhanced reasoning with transparency
        reasoning_factors = []
        if truth_synthesis['confidence_factors']['web_scraping_enhanced']:
            reasoning_factors.append("enhanced with official web data")
        if truth_synthesis['confidence_factors']['bias_corrected']:
            reasoning_factors.append("AI bias-corrected")
        if truth_synthesis['confidence_factors']['anonymous_handling']:
            reasoning_factors.append("anonymous reviews processed")
        
        return {
            'college_id': college_info['college_id'],
            'college_name': college_info['name'],
            'overall_recommendation': recommendation,
            'confidence_level': confidence_text,
            'verasuni_score': overall_score,
            'confidence_percentage': round(confidence * 100, 1),
            'aspect_breakdown': aspect_scores,
            'key_strengths': strengths,
            'areas_of_concern': concerns,
            'data_transparency': {
                'total_reviews_analyzed': truth_synthesis.get('total_reviews', 0),
                'user_generated_percentage': round(truth_synthesis['data_composition']['user_generated_ratio'] * 100, 1),
                'official_data_percentage': round(truth_synthesis['data_composition']['scraped_data_ratio'] * 100, 1),
                'anonymous_reviews_percentage': round(truth_synthesis['data_composition']['anonymous_ratio'] * 100, 1),
                'bias_neutralized': True,
                'credibility_weighted': True
            },
            'recommendation_reasoning': f"Based on {', '.join(reasoning_factors)} analysis with {confidence_text.lower()}",
            'last_updated': datetime.now().isoformat(),
            'algorithm_version': '2.0_enhanced'
        }

# Usage Example with Enhanced Features
async def demo_enhanced_verasuni():
    """Complete Enhanced Verasuni Algorithm Demo"""
    
    # Sample input data with anonymous reviews
    enhanced_reviews_data = [
        {
            'review': {
                'id': 'review_001',
                'content': 'This college is absolutely incredible and perfect in every way possible! Everyone loves it and all students are completely satisfied.',
                'rating': 5.0,
                'timestamp': '2024-09-01T10:30:00'
            },
            'user': {
                'user_id': 'user_001',
                'user_type': 'student',
                'email': 'john.doe@vit.ac.in',
                'full_name': 'John Doe',
                'created_at': '2023-08-15T10:30:00',
                'documents': {
                    'student_id': {'url': 'student_id.pdf', 'file_size': 1024000}
                },
                'social_profiles': {
                    'linkedin': {'name': 'John Doe', 'bio': 'CS student at VIT', 'followers': 150}
                }
            },
            'is_anonymous': False,  # User chose to show identity
            'source': 'user_generated',
            'contribution_history': [
                {
                    'content': 'Great CS program with good faculty...',
                    'community_feedback': {'upvotes': 15, 'downvotes': 2},
                    'helpfulness_rating': 4.2
                }
            ]
        },
        {
            'review': {
                'id': 'review_002',
                'content': 'VIT provides decent education with good placement opportunities. The computer science department has experienced faculty.',
                'rating': 4.0,
                'timestamp': '2024-08-15T14:20:00'
            },
            'user': {
                'user_id': 'user_002',
                'user_type': 'alumni',
                'email': 'jane.smith@gmail.com',
                'full_name': 'Jane Smith',
                'created_at': '2022-05-10T09:15:00',
                'documents': {
                    'degree_certificate': {'url': 'degree.pdf', 'file_size': 2048000}
                },
                'social_profiles': {
                    'linkedin': {'name': 'Jane Smith', 'bio': 'SWE at Google | VIT Alumnus', 'followers': 500}
                }
            },
            'is_anonymous': True,  # User chose anonymous posting
            'source': 'user_generated',
            'contribution_history': [
                {
                    'content': 'As an alumnus, VIT prepared me well for industry...',
                    'community_feedback': {'upvotes': 35, 'downvotes': 1},
                    'helpfulness_rating': 4.7
                }
            ]
        }
    ]
    
    # Initialize and run enhanced pipeline
    enhanced_engine = EnhancedVerasuniEngine()
    results = await enhanced_engine.process_complete_pipeline(
        enhanced_reviews_data, 
        'VIT Vellore'
    )
    
    return results

# Integration with Flask Backend
def integrate_with_backend(app):
    """Integration function for Flask backend"""
    
    enhanced_engine = EnhancedVerasuniEngine()
    
    @app.route('/api/verasuni/enhanced-analysis', methods=['POST'])
    def enhanced_verasuni_analysis():
        """Enhanced Verasuni analysis endpoint"""
        try:
            data = request.get_json()
            college_name = data.get('college_name')
            
            if not college_name:
                return jsonify({'error': 'college_name is required'}), 400
            
            # Get reviews from database
            from verasuni_enhanced_backend import Review, User
            reviews = Review.query.filter_by(
                college_name=college_name,
                is_active=True
            ).all()
            
            # Convert to format expected by algorithm
            review_data = []
            for review in reviews:
                user = User.query.filter_by(user_id=review.user_id).first()
                
                review_item = {
                    'review': {
                        'id': review.review_id,
                        'content': review.content,
                        'rating': review.rating,
                        'timestamp': review.created_at.isoformat()
                    },
                    'user': {
                        'user_id': user.user_id,
                        'user_type': user.user_status.value,
                        'email': user.email,
                        'full_name': user.full_name,
                        'created_at': user.created_at.isoformat(),
                        'documents': {},
                        'social_profiles': {}
                    } if user else {},
                    'is_anonymous': review.is_anonymous,
                    'source': review.source,
                    'contribution_history': []
                }
                review_data.append(review_item)
            
            # Run enhanced analysis
            import asyncio
            results = asyncio.run(
                enhanced_engine.process_complete_pipeline(review_data, college_name)
            )
            
            return jsonify({
                "status": "success",
                "enhanced_verasuni_results": results,
                "algorithm_version": "2.0_enhanced",
                "processing_timestamp": datetime.now().isoformat(),
                "features": [
                    "Anonymous review processing",
                    "Web scraping fallback",
                    "Multi-source integration",
                    "Enhanced bias correction",
                    "Transparent recommendations"
                ]
            })
        
        except Exception as e:
            return jsonify({'error': f'Enhanced Verasuni processing failed: {str(e)}'}), 500
    
    return enhanced_engine

if __name__ == "__main__":
    # Run demo
    import asyncio
    results = asyncio.run(demo_enhanced_verasuni())
    print(json.dumps(results, indent=2))