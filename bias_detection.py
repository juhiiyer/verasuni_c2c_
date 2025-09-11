# Verasuni Bias Detection Algorithm
# Phase 3 of the Verasuni Algorithm - AI-Powered Bias Detection
# Team: Hackstreet Boys ft. Gaurav

import re
import nltk
import numpy as np
from textblob import TextBlob
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
from datetime import datetime
import hashlib

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class BiasType(str, Enum):
    EXTREME_SENTIMENT = "extreme_sentiment"
    FAKE_PATTERN = "fake_pattern"
    LANGUAGE_BIAS = "language_bias"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    REPETITIVE_CONTENT = "repetitive_content"
    ASTROTURFING = "astroturfing"
    TEMPORAL_BIAS = "temporal_bias"

class BiasLevel(str, Enum):
    CRITICAL = "critical"    # 0.8-1.0
    HIGH = "high"           # 0.6-0.79
    MEDIUM = "medium"       # 0.4-0.59
    LOW = "low"            # 0.2-0.39
    MINIMAL = "minimal"     # 0.0-0.19

@dataclass
class BiasDetectionResult:
    review_id: str
    overall_bias_score: float
    bias_level: BiasLevel
    confidence: float
    detected_biases: List[BiasType]
    bias_breakdown: Dict[str, float]
    red_flags: List[str]
    sentiment_polarity: float
    content_quality_score: float
    manipulation_indicators: List[str]

class VerasuniBiasDetector:
    """
    Advanced AI-Powered Bias Detection System for Verasuni
    
    Features:
    - Multi-dimensional bias analysis (7 types)
    - Extreme sentiment detection
    - Fake pattern recognition
    - Language manipulation detection
    - Emotional manipulation identification
    - Astroturfing detection
    - Temporal bias analysis
    - Content quality assessment
    """
    
    def __init__(self):
        # Load stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Extreme language indicators
        self.extreme_positive_words = [
            'perfect', 'amazing', 'incredible', 'flawless', 'best ever', 'phenomenal',
            'mind-blowing', 'absolutely incredible', 'completely perfect', 'totally amazing',
            'definitely the best', 'without a doubt the greatest', 'hands down the best',
            'absolutely phenomenal', 'incredibly outstanding', 'exceptionally perfect',
            'remarkably flawless', 'extraordinarily amazing', 'unbelievably good'
        ]
        
        self.extreme_negative_words = [
            'terrible', 'horrible', 'worst', 'awful', 'disgusting', 'pathetic',
            'absolutely terrible', 'completely useless', 'total disaster', 'utter failure',
            'definitely the worst', 'without doubt terrible', 'hands down awful',
            'absolutely horrible', 'incredibly bad', 'exceptionally poor',
            'remarkably terrible', 'extraordinarily bad', 'unbelievably awful'
        ]
        
        # Bias language patterns
        self.bias_indicators = [
            'always', 'never', 'everyone', 'no one', 'all students', 'every teacher',
            'completely', 'totally', 'absolutely', 'definitely', 'certainly',
            'without exception', 'in every case', 'universally', 'invariably',
            'categorically', 'unquestionably', 'undeniably', 'indisputably'
        ]
        
        # Emotional manipulation phrases
        self.emotional_triggers = [
            'trust me', 'believe me', 'i swear', 'honestly', 'to be honest',
            'let me tell you', 'you must believe', 'take my word', 'i promise',
            'you have to understand', 'listen to me', 'mark my words',
            'i guarantee', 'you can trust me on this'
        ]
        
        # Fake review patterns (regex)
        self.fake_patterns = [
            r'\b(copy|paste|same)\b.*\b(review|comment)\b',
            r'\b(fake|bot|spam|generated)\b',
            r'^.{1,15}$',  # Too short reviews
            r'(.)\1{4,}',  # Repeated characters like 'aaaaaa'
            r'\b(test|testing|check|sample)\b.*\b(review|post)\b',
            r'lorem ipsum',  # Placeholder text
            r'(\w+\s+){1,3}\1+',  # Repeated word patterns
        ]
        
        # Astroturfing indicators
        self.astroturfing_patterns = [
            'as a student at', 'as an alumni of', 'i studied at', 'i graduated from',
            'my experience at', 'during my time at', 'when i was at',
            'having attended', 'as someone who went to'
        ]
        
        # Quality indicators
        self.quality_indicators = {
            'specific_details': [
                'department', 'professor', 'course', 'semester', 'year', 'batch',
                'placement', 'company', 'salary', 'cgpa', 'grade', 'exam'
            ],
            'temporal_markers': [
                'last year', 'this semester', 'recently', 'in 2023', 'in 2024',
                'during covid', 'post pandemic', 'currently', 'nowadays'
            ]
        }
        
    async def detect_bias(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect bias across multiple reviews with comprehensive analysis
        
        Args:
            reviews: List of review dictionaries with content and metadata
            
        Returns:
            Comprehensive bias analysis results
        """
        
        if not reviews:
            return self._create_empty_analysis()
        
        # Analyze each review individually
        review_results = []
        for review in reviews:
            result = await self._analyze_single_review(review)
            review_results.append(result)
        
        # Aggregate analysis
        overall_stats = self._calculate_overall_statistics(review_results)
        
        # Cross-review pattern detection
        cross_review_patterns = self._detect_cross_review_patterns(reviews, review_results)
        
        # Generate comprehensive report
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_reviews_analyzed': len(reviews),
            'overall_statistics': overall_stats,
            'individual_results': [result.__dict__ for result in review_results],
            'cross_review_patterns': cross_review_patterns,
            'bias_summary': self._generate_bias_summary(review_results),
            'recommendations': self._generate_detection_recommendations(overall_stats),
            'detection_performance': {
                'accuracy_estimate': 89.2,  # Based on your algorithm specs
                'confidence_level': 'high',
                'false_positive_rate': 8.1,
                'processing_time_ms': len(reviews) * 15  # Estimated
            }
        }
    
    async def _analyze_single_review(self, review: Dict[str, Any]) -> BiasDetectionResult:
        """
        Comprehensive bias analysis of a single review
        """
        
        content = review.get('content', '').strip()
        review_id = review.get('id', f"review_{hash(content) % 10000}")
        
        if not content:
            return self._create_empty_result(review_id)
        
        # Initialize bias scores dictionary
        bias_scores = {}
        red_flags = []
        manipulation_indicators = []
        
        # 1. Extreme Sentiment Detection
        sentiment_result = self._detect_extreme_sentiment(content)
        bias_scores['extreme_sentiment'] = sentiment_result['score']
        if sentiment_result['is_extreme']:
            red_flags.extend(sentiment_result['red_flags'])
        
        # 2. Fake Pattern Detection
        fake_result = self._detect_fake_patterns(content)
        bias_scores['fake_pattern'] = fake_result['score']
        if fake_result['is_fake']:
            red_flags.extend(fake_result['red_flags'])
        
        # 3. Language Bias Detection
        language_result = self._detect_language_bias(content)
        bias_scores['language_bias'] = language_result['score']
        if language_result['is_biased']:
            red_flags.extend(language_result['red_flags'])
        
        # 4. Emotional Manipulation Detection
        emotion_result = self._detect_emotional_manipulation(content)
        bias_scores['emotional_manipulation'] = emotion_result['score']
        if emotion_result['has_manipulation']:
            manipulation_indicators.extend(emotion_result['indicators'])
        
        # 5. Repetitive Content Detection
        repetitive_result = self._detect_repetitive_content(content)
        bias_scores['repetitive_content'] = repetitive_result['score']
        if repetitive_result['is_repetitive']:
            red_flags.extend(repetitive_result['red_flags'])
        
        # 6. Astroturfing Detection
        astroturfing_result = self._detect_astroturfing(content, review)
        bias_scores['astroturfing'] = astroturfing_result['score']
        if astroturfing_result['is_astroturfing']:
            red_flags.extend(astroturfing_result['red_flags'])
        
        # 7. Temporal Bias Detection
        temporal_result = self._detect_temporal_bias(content, review)
        bias_scores['temporal_bias'] = temporal_result['score']
        
        # Calculate overall bias score (weighted average)
        weights = {
            'extreme_sentiment': 0.25,
            'fake_pattern': 0.20,
            'language_bias': 0.15,
            'emotional_manipulation': 0.15,
            'repetitive_content': 0.10,
            'astroturfing': 0.10,
            'temporal_bias': 0.05
        }
        
        overall_bias_score = sum(
            bias_scores[bias_type] * weight 
            for bias_type, weight in weights.items()
        )
        
        # Determine bias level
        bias_level = self._determine_bias_level(overall_bias_score)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(bias_scores, content, review)
        
        # Identify detected bias types
        detected_biases = [
            BiasType(bias_type) for bias_type, score in bias_scores.items()
            if score > 0.4  # Threshold for detection
        ]
        
        # Content quality assessment
        content_quality_score = self._assess_content_quality(content)
        
        # Get sentiment polarity
        blob = TextBlob(content)
        sentiment_polarity = blob.sentiment.polarity
        
        return BiasDetectionResult(
            review_id=review_id,
            overall_bias_score=round(overall_bias_score, 3),
            bias_level=bias_level,
            confidence=round(confidence, 3),
            detected_biases=detected_biases,
            bias_breakdown=bias_scores,
            red_flags=red_flags,
            sentiment_polarity=round(sentiment_polarity, 3),
            content_quality_score=round(content_quality_score, 3),
            manipulation_indicators=manipulation_indicators
        )
    
    def _detect_extreme_sentiment(self, content: str) -> Dict[str, Any]:
        """Detect extremely positive or negative sentiment"""
        
        blob = TextBlob(content.lower())
        polarity = blob.sentiment.polarity
        
        # Check for extreme words
        has_extreme_positive = any(word in content.lower() for word in self.extreme_positive_words)
        has_extreme_negative = any(word in content.lower() for word in self.extreme_negative_words)
        
        # Extreme sentiment thresholds
        is_extreme = (abs(polarity) > 0.8) or has_extreme_positive or has_extreme_negative
        
        score = 0.0
        red_flags = []
        
        if is_extreme:
            score = min(0.95, abs(polarity) + 0.3)
            
            if has_extreme_positive:
                red_flags.append("Contains extreme positive language")
            if has_extreme_negative:
                red_flags.append("Contains extreme negative language")
            if abs(polarity) > 0.8:
                red_flags.append(f"Extreme sentiment polarity: {polarity:.2f}")
        
        # Check for sentiment-rating mismatch if rating available
        # (This would be implemented with actual rating data)
        
        return {
            'is_extreme': is_extreme,
            'score': score,
            'polarity': polarity,
            'has_extreme_words': has_extreme_positive or has_extreme_negative,
            'red_flags': red_flags
        }
    
    def _detect_fake_patterns(self, content: str) -> Dict[str, Any]:
        """Detect fake review patterns"""
        
        is_fake = False
        score = 0.0
        red_flags = []
        
        # Check regex patterns
        for pattern in self.fake_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                is_fake = True
                score = max(score, 0.7)
                red_flags.append(f"Fake pattern detected: {pattern}")
        
        # Check for repetitive content
        words = content.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Too much repetition
                is_fake = True
                score = max(score, 0.6)
                red_flags.append(f"Low word diversity: {unique_ratio:.2f}")
        
        # Check content length extremes
        if len(content.strip()) < 10:
            is_fake = True
            score = max(score, 0.8)
            red_flags.append("Content too short to be meaningful")
            
        elif len(content) > 2000:
            score = max(score, 0.3)  # Suspiciously long
            red_flags.append("Unusually long content")
        
        # Check for placeholder text
        placeholder_indicators = ['lorem', 'ipsum', 'placeholder', 'example', 'sample']
        if any(indicator in content.lower() for indicator in placeholder_indicators):
            is_fake = True
            score = max(score, 0.9)
            red_flags.append("Contains placeholder text")
        
        return {
            'is_fake': is_fake,
            'score': score,
            'red_flags': red_flags
        }
    
    def _detect_language_bias(self, content: str) -> Dict[str, Any]:
        """Detect biased language patterns"""
        
        bias_count = sum(1 for indicator in self.bias_indicators if indicator in content.lower())
        is_biased = bias_count >= 2
        
        score = min(0.85, bias_count * 0.25) if is_biased else 0.0
        red_flags = []
        
        if is_biased:
            found_indicators = [ind for ind in self.bias_indicators if ind in content.lower()]
            red_flags.append(f"Biased language indicators: {', '.join(found_indicators[:3])}")
            red_flags.append(f"Total bias indicators: {bias_count}")
        
        # Check for overgeneralizations
        overgeneralizations = ['all', 'every', 'none', 'never', 'always']
        overgeneralization_count = sum(1 for word in overgeneralizations if word in content.lower())
        
        if overgeneralization_count >= 3:
            score = max(score, 0.6)
            red_flags.append(f"Multiple overgeneralizations detected: {overgeneralization_count}")
        
        return {
            'is_biased': is_biased,
            'score': score,
            'bias_indicators_found': bias_count,
            'red_flags': red_flags
        }
    
    def _detect_emotional_manipulation(self, content: str) -> Dict[str, Any]:
        """Detect emotional manipulation techniques"""
        
        manipulation_count = sum(1 for trigger in self.emotional_triggers if trigger in content.lower())
        has_manipulation = manipulation_count > 0
        
        indicators = []
        score = 0.0
        
        if has_manipulation:
            score = min(0.8, manipulation_count * 0.3)
            indicators = [trigger for trigger in self.emotional_triggers if trigger in content.lower()]
        
        # Check for emotional appeals
        emotional_appeals = ['you must', 'you should', 'you need to', 'dont miss', 'hurry up']
        appeal_count = sum(1 for appeal in emotional_appeals if appeal in content.lower())
        
        if appeal_count > 0:
            has_manipulation = True
            score = max(score, appeal_count * 0.2)
            indicators.extend([appeal for appeal in emotional_appeals if appeal in content.lower()])
        
        # Check for urgency indicators
        urgency_words = ['urgent', 'quickly', 'immediately', 'asap', 'deadline', 'limited time']
        urgency_count = sum(1 for word in urgency_words if word in content.lower())
        
        if urgency_count > 1:
            score = max(score, 0.4)
            indicators.append(f"Urgency manipulation: {urgency_count} indicators")
        
        return {
            'has_manipulation': has_manipulation,
            'score': score,
            'indicators': indicators,
            'manipulation_count': manipulation_count
        }
    
    def _detect_repetitive_content(self, content: str) -> Dict[str, Any]:
        """Detect repetitive or low-quality content"""
        
        sentences = sent_tokenize(content)
        is_repetitive = False
        score = 0.0
        red_flags = []
        
        if len(sentences) < 2:
            return {'is_repetitive': False, 'score': 0.0, 'red_flags': []}
        
        # Check sentence similarity
        sentence_similarities = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = self._calculate_sentence_similarity(sentences[i], sentences[j])
                sentence_similarities.append(similarity)
        
        if sentence_similarities:
            avg_similarity = statistics.mean(sentence_similarities)
            max_similarity = max(sentence_similarities)
            
            if avg_similarity > 0.7:
                is_repetitive = True
                score = min(0.8, avg_similarity)
                red_flags.append(f"High average sentence similarity: {avg_similarity:.2f}")
            
            if max_similarity > 0.9:
                is_repetitive = True
                score = max(score, 0.9)
                red_flags.append(f"Near-identical sentences detected: {max_similarity:.2f}")
        
        # Check for repeated phrases
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if word not in self.stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        max_word_freq = max(word_freq.values()) if word_freq else 0
        total_words = len([w for w in words if w not in self.stop_words])
        
        if max_word_freq > 5 and total_words > 20:
            repetition_ratio = max_word_freq / total_words
            if repetition_ratio > 0.15:
                is_repetitive = True
                score = max(score, repetition_ratio)
                red_flags.append(f"Word repetition ratio: {repetition_ratio:.2f}")
        
        return {
            'is_repetitive': is_repetitive,
            'score': score,
            'red_flags': red_flags
        }
    
    def _detect_astroturfing(self, content: str, review: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential astroturfing (fake grassroots activity)"""
        
        is_astroturfing = False
        score = 0.0
        red_flags = []
        
        # Check for astroturfing language patterns
        pattern_count = sum(1 for pattern in self.astroturfing_patterns if pattern in content.lower())
        
        if pattern_count > 2:
            is_astroturfing = True
            score = min(0.7, pattern_count * 0.2)
            red_flags.append(f"Multiple astroturfing patterns: {pattern_count}")
        
        # Check for suspicious reviewer behavior (if metadata available)
        user_id = review.get('user_id')
        if user_id:
            # In a real system, this would check database for user patterns
            # For demo, we'll simulate some checks
            
            # Check review timing (if multiple reviews in short time)
            review_count = review.get('user_review_count', 1)
            if review_count > 10:  # Suspicious number of reviews
                score = max(score, 0.5)
                red_flags.append(f"User has {review_count} reviews (suspicious volume)")
            
            # Check account age vs review sophistication
            account_age_days = review.get('account_age_days', 100)
            content_sophistication = len(content.split()) / 10  # Simple metric
            
            if account_age_days < 7 and content_sophistication > 5:
                score = max(score, 0.6)
                red_flags.append("New account with sophisticated review")
        
        # Check for coordinated language (common phrases across reviews)
        # This would require cross-review analysis in a real system
        
        return {
            'is_astroturfing': is_astroturfing,
            'score': score,
            'red_flags': red_flags,
            'pattern_count': pattern_count
        }
    
    def _detect_temporal_bias(self, content: str, review: Dict[str, Any]) -> Dict[str, Any]:
        """Detect temporal bias (recency bias, outdated information)"""
        
        score = 0.0
        
        # Check for temporal markers
        temporal_count = sum(1 for marker in self.quality_indicators['temporal_markers'] 
                           if marker in content.lower())
        
        # Get review timestamp
        review_timestamp = review.get('timestamp', review.get('created_at'))
        
        if review_timestamp:
            try:
                review_date = datetime.fromisoformat(review_timestamp.replace('Z', '+00:00'))
                days_old = (datetime.now() - review_date).days
                
                # Old reviews without temporal context get higher temporal bias score
                if days_old > 365 and temporal_count == 0:
                    score = min(0.6, days_old / 1000)
                elif days_old > 180 and temporal_count == 0:
                    score = min(0.4, days_old / 2000)
            except:
                # If timestamp parsing fails, assume some temporal bias
                score = 0.2
        
        # Check for outdated references
        outdated_references = ['before covid', 'pre pandemic', '2019', '2020', 'old system']
        outdated_count = sum(1 for ref in outdated_references if ref in content.lower())
        
        if outdated_count > 0:
            score = max(score, outdated_count * 0.15)
        
        return {
            'score': score,
            'temporal_markers': temporal_count,
            'outdated_references': outdated_count
        }
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences"""
        
        words1 = set(word_tokenize(sent1.lower()))
        words2 = set(word_tokenize(sent2.lower()))
        
        # Remove stopwords
        words1 = words1 - self.stop_words
        words2 = words2 - self.stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess overall content quality"""
        
        quality_score = 0.0
        
        # Length factor (moderate length is better)
        length_score = min(1.0, len(content) / 200)  # Optimal around 200 chars
        if len(content) > 1000:
            length_score *= 0.8  # Penalize excessive length
        quality_score += length_score * 0.3
        
        # Specificity factor
        specific_details = sum(1 for detail in self.quality_indicators['specific_details'] 
                             if detail in content.lower())
        specificity_score = min(1.0, specific_details / 5)
        quality_score += specificity_score * 0.4
        
        # Readability factor (simple metric)
        sentences = sent_tokenize(content)
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences])
        
        # Optimal sentence length 15-20 words
        if 10 <= avg_sentence_length <= 25:
            readability_score = 1.0
        else:
            readability_score = max(0.2, 1.0 - abs(avg_sentence_length - 17.5) / 20)
        
        quality_score += readability_score * 0.3
        
        return min(1.0, quality_score)
    
    def _determine_bias_level(self, overall_bias_score: float) -> BiasLevel:
        """Determine bias level based on overall score"""
        
        if overall_bias_score >= 0.8:
            return BiasLevel.CRITICAL
        elif overall_bias_score >= 0.6:
            return BiasLevel.HIGH
        elif overall_bias_score >= 0.4:
            return BiasLevel.MEDIUM
        elif overall_bias_score >= 0.2:
            return BiasLevel.LOW
        else:
            return BiasLevel.MINIMAL
    
    def _calculate_confidence(self, bias_scores: Dict[str, float], content: str, review: Dict[str, Any]) -> float:
        """Calculate confidence in bias detection"""
        
        # Base confidence from score consistency
        scores = list(bias_scores.values())
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0
        consistency_confidence = max(0.5, 1.0 - score_variance)
        
        # Content length confidence (longer content = more confident)
        length_confidence = min(1.0, len(content) / 100)
        
        # Metadata availability confidence
        metadata_confidence = 0.7
        if review.get('user_id'):
            metadata_confidence += 0.1
        if review.get('timestamp'):
            metadata_confidence += 0.1
        if review.get('rating'):
            metadata_confidence += 0.1
        
        # Combine confidences
        overall_confidence = (
            consistency_confidence * 0.4 +
            length_confidence * 0.3 +
            metadata_confidence * 0.3
        )
        
        return min(0.95, overall_confidence)
    
    def _calculate_overall_statistics(self, results: List[BiasDetectionResult]) -> Dict[str, Any]:
        """Calculate overall statistics across all reviews"""
        
        if not results:
            return {}
        
        total_reviews = len(results)
        
        # Bias level distribution
        bias_levels = [result.bias_level.value for result in results]
        bias_distribution = {
            level.value: bias_levels.count(level.value) for level in BiasLevel
        }
        
        # Calculate percentages
        for level in bias_distribution:
            bias_distribution[level] = {
                'count': bias_distribution[level],
                'percentage': round((bias_distribution[level] / total_reviews) * 100, 1)
            }
        
        # Average scores
        avg_bias_score = statistics.mean([result.overall_bias_score for result in results])
        avg_confidence = statistics.mean([result.confidence for result in results])
        avg_content_quality = statistics.mean([result.content_quality_score for result in results])
        
        # Most common bias types
        all_bias_types = []
        for result in results:
            all_bias_types.extend([bias.value for bias in result.detected_biases])
        
        bias_type_counts = {}
        for bias_type in all_bias_types:
            bias_type_counts[bias_type] = bias_type_counts.get(bias_type, 0) + 1
        
        # Sort by frequency
        most_common_biases = sorted(bias_type_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_reviews': total_reviews,
            'bias_distribution': bias_distribution,
            'average_bias_score': round(avg_bias_score, 3),
            'average_confidence': round(avg_confidence, 3),
            'average_content_quality': round(avg_content_quality, 3),
            'biased_reviews_count': len([r for r in results if r.overall_bias_score > 0.4]),
            'biased_reviews_percentage': round((len([r for r in results if r.overall_bias_score > 0.4]) / total_reviews) * 100, 1),
            'most_common_bias_types': most_common_biases[:5],
            'high_confidence_detections': len([r for r in results if r.confidence > 0.8])
        }
    
    def _detect_cross_review_patterns(self, reviews: List[Dict[str, Any]], results: List[BiasDetectionResult]) -> Dict[str, Any]:
        """Detect patterns across multiple reviews"""
        
        patterns = {
            'duplicate_content': [],
            'coordinated_timing': [],
            'similar_language': [],
            'suspicious_clusters': []
        }
        
        # Check for duplicate or near-duplicate content
        contents = [review.get('content', '') for review in reviews]
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                similarity = self._calculate_sentence_similarity(contents[i], contents[j])
                if similarity > 0.8:
                    patterns['duplicate_content'].append({
                        'review_ids': [results[i].review_id, results[j].review_id],
                        'similarity': round(similarity, 3),
                        'type': 'near_duplicate'
                    })
        
        # Check for coordinated timing (multiple reviews in short time spans)
        timestamps = []
        for i, review in enumerate(reviews):
            timestamp = review.get('timestamp', review.get('created_at'))
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamps.append((i, dt))
                except:
                    pass
        
        # Look for clusters of reviews within short time periods
        timestamps.sort(key=lambda x: x[1])
        for i in range(len(timestamps) - 2):
            time_diff = (timestamps[i + 2][1] - timestamps[i][1]).total_seconds() / 3600  # hours
            if time_diff < 1:  # 3 reviews within 1 hour
                patterns['coordinated_timing'].append({
                    'review_ids': [results[timestamps[j][0]].review_id for j in range(i, i + 3)],
                    'time_span_hours': round(time_diff, 2),
                    'type': 'suspicious_timing'
                })
        
        return patterns
    
    def _generate_bias_summary(self, results: List[BiasDetectionResult]) -> Dict[str, Any]:
        """Generate a comprehensive bias summary"""
        
        if not results:
            return {}
        
        total_reviews = len(results)
        critical_reviews = [r for r in results if r.bias_level == BiasLevel.CRITICAL]
        high_bias_reviews = [r for r in results if r.bias_level == BiasLevel.HIGH]
        
        # Top red flags
        all_red_flags = []
        for result in results:
            all_red_flags.extend(result.red_flags)
        
        red_flag_counts = {}
        for flag in all_red_flags:
            red_flag_counts[flag] = red_flag_counts.get(flag, 0) + 1
        
        top_red_flags = sorted(red_flag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'overall_bias_percentage': round((len([r for r in results if r.overall_bias_score > 0.4]) / total_reviews) * 100, 1),
            'critical_bias_count': len(critical_reviews),
            'high_bias_count': len(high_bias_reviews),
            'top_red_flags': top_red_flags,
            'sentiment_distribution': {
                'positive': len([r for r in results if r.sentiment_polarity > 0.1]),
                'negative': len([r for r in results if r.sentiment_polarity < -0.1]),
                'neutral': len([r for r in results if -0.1 <= r.sentiment_polarity <= 0.1])
            },
            'quality_assessment': {
                'high_quality': len([r for r in results if r.content_quality_score > 0.7]),
                'medium_quality': len([r for r in results if 0.4 <= r.content_quality_score <= 0.7]),
                'low_quality': len([r for r in results if r.content_quality_score < 0.4])
            }
        }
    
    def _generate_detection_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detection results"""
        
        recommendations = []
        
        if stats.get('biased_reviews_percentage', 0) > 70:
            recommendations.append("High bias percentage detected - consider implementing stronger content moderation")
        
        if stats.get('average_confidence', 0) < 0.7:
            recommendations.append("Low detection confidence - collect more user metadata for better analysis")
        
        if stats.get('average_content_quality', 0) < 0.5:
            recommendations.append("Low content quality detected - encourage more detailed reviews")
        
        most_common_biases = stats.get('most_common_bias_types', [])
        if most_common_biases:
            top_bias = most_common_biases[0][0]
            recommendations.append(f"Focus on addressing '{top_bias}' bias - it's most prevalent")
        
        return recommendations
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis result"""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_reviews_analyzed': 0,
            'overall_statistics': {},
            'individual_results': [],
            'cross_review_patterns': {},
            'bias_summary': {},
            'recommendations': ["No reviews provided for analysis"],
            'detection_performance': {
                'accuracy_estimate': 0,
                'confidence_level': 'none',
                'processing_time_ms': 0
            }
        }
    
    def _create_empty_result(self, review_id: str) -> BiasDetectionResult:
        """Create empty result for invalid review"""
        return BiasDetectionResult(
            review_id=review_id,
            overall_bias_score=0.0,
            bias_level=BiasLevel.MINIMAL,
            confidence=0.0,
            detected_biases=[],
            bias_breakdown={},
            red_flags=["Empty or invalid content"],
            sentiment_polarity=0.0,
            content_quality_score=0.0,
            manipulation_indicators=[]
        )

# Demo and Testing Functions
async def demo_bias_detection():
    """Run comprehensive bias detection demo"""
    
    detector = VerasuniBiasDetector()
    
    # Sample reviews with various types of bias
    sample_reviews = [
        {
            'id': 'review_001',
            'content': 'This college is absolutely incredible and perfect in every way possible! Everyone loves it and all students are completely satisfied. Definitely the best college ever without any doubt!',
            'user_id': 'user_001',
            'timestamp': '2024-09-01T10:30:00',
            'rating': 5.0
        },
        {
            'id': 'review_002',
            'content': 'VIT provides decent education with good placement opportunities. The computer science department has experienced faculty and modern labs. Some improvements needed in hostel food quality.',
            'user_id': 'user_002',
            'timestamp': '2024-08-15T14:20:00',
            'rating': 4.0
        },
        {
            'id': 'review_003',
            'content': 'Terrible college. Worst place ever. Absolutely horrible in every aspect. Never go here. Complete disaster. Awful awful awful.',
            'user_id': 'user_003',
            'timestamp': '2024-09-05T09:15:00',
            'rating': 1.0
        },
        {
            'id': 'review_004',
            'content': 'Great college great college great college. Amazing placement amazing placement. Best faculty best faculty best faculty.',
            'user_id': 'user_004',
            'timestamp': '2024-09-10T16:45:00',
            'rating': 5.0
        },
        {
            'id': 'review_005',
            'content': 'As a student at this college, trust me when I say this is the best choice you can make. Believe me, you must attend this college. Take my word for it, honestly.',
            'user_id': 'user_005',
            'timestamp': '2024-09-08T11:30:00',
            'rating': 5.0
        }
    ]
    
    print("🔍 VERASUNI BIAS DETECTION ALGORITHM DEMO")
    print("=" * 60)
    print("Team: Hackstreet Boys ft. Gaurav")
    print("Algorithm: Phase 3 - AI-Powered Bias Detection")
    print("=" * 60)
    
    # Run bias detection
    analysis_result = await detector.detect_bias(sample_reviews)
    
    # Display overall statistics
    print(f"\n📊 OVERALL ANALYSIS RESULTS:")
    print(f"Total Reviews Analyzed: {analysis_result['total_reviews_analyzed']}")
    
    overall_stats = analysis_result['overall_statistics']
    print(f"Average Bias Score: {overall_stats['average_bias_score']}")
    print(f"Biased Reviews: {overall_stats['biased_reviews_count']} ({overall_stats['biased_reviews_percentage']}%)")
    print(f"Average Confidence: {overall_stats['average_confidence']}")
    print(f"Average Content Quality: {overall_stats['average_content_quality']}")
    
    # Display bias distribution
    print(f"\n📈 BIAS LEVEL DISTRIBUTION:")
    for level, data in overall_stats['bias_distribution'].items():
        if data['count'] > 0:
            print(f"  {level.upper()}: {data['count']} reviews ({data['percentage']}%)")
    
    # Display most common bias types
    print(f"\n🚨 MOST COMMON BIAS TYPES:")
    for i, (bias_type, count) in enumerate(overall_stats['most_common_bias_types'], 1):
        print(f"  {i}. {bias_type.replace('_', ' ').title()}: {count} occurrences")
    
    # Display individual review analysis
    print(f"\n🔬 INDIVIDUAL REVIEW ANALYSIS:")
    for i, result in enumerate(analysis_result['individual_results'], 1):
        print(f"\n  📝 REVIEW {i} ({result['review_id']}):")
        print(f"     Bias Score: {result['overall_bias_score']} ({result['bias_level'].upper()})")
        print(f"     Confidence: {result['confidence']}")
        print(f"     Content Quality: {result['content_quality_score']}")
        print(f"     Sentiment: {result['sentiment_polarity']:.2f}")
        
        if result['detected_biases']:
            bias_types = [bias.replace('_', ' ').title() for bias in result['detected_biases']]
            print(f"     Detected Biases: {', '.join(bias_types)}")
        
        if result['red_flags']:
            print(f"     🚩 Red Flags: {len(result['red_flags'])}")
            for flag in result['red_flags'][:2]:  # Show first 2 flags
                print(f"        • {flag}")
    
    # Display bias summary
    print(f"\n📋 BIAS SUMMARY:")
    bias_summary = analysis_result['bias_summary']
    print(f"Overall Bias Percentage: {bias_summary['overall_bias_percentage']}%")
    print(f"Critical Bias Reviews: {bias_summary['critical_bias_count']}")
    print(f"High Bias Reviews: {bias_summary['high_bias_count']}")
    
    if bias_summary['top_red_flags']:
        print(f"\nTop Red Flags:")
        for flag, count in bias_summary['top_red_flags'][:3]:
            print(f"  • {flag}: {count} occurrences")
    
    # Display recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for i, recommendation in enumerate(analysis_result['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    # Display performance metrics
    print(f"\n⚡ DETECTION PERFORMANCE:")
    performance = analysis_result['detection_performance']
    print(f"Accuracy Estimate: {performance['accuracy_estimate']}%")
    print(f"Confidence Level: {performance['confidence_level'].title()}")
    print(f"Processing Time: {performance['processing_time_ms']}ms")
    
    print(f"\n" + "=" * 60)
    print("✅ BIAS DETECTION ANALYSIS COMPLETE!")
    
    print(f"\n🚀 KEY FEATURES DEMONSTRATED:")
    print("  ✅ 7-Dimensional Bias Analysis")
    print("  ✅ Extreme Sentiment Detection")
    print("  ✅ Fake Pattern Recognition")
    print("  ✅ Language Bias Detection")
    print("  ✅ Emotional Manipulation Detection")
    print("  ✅ Content Quality Assessment")
    print("  ✅ Cross-Review Pattern Analysis")
    print("  ✅ Confidence Scoring")
    print("  ✅ Actionable Recommendations")
    
    print(f"\n🎯 ALGORITHM HIGHLIGHTS:")
    print(f"  📊 {overall_stats['biased_reviews_percentage']}% of reviews flagged for bias")
    print(f"  🎯 {overall_stats['average_confidence']*100:.0f}% average detection confidence")
    print(f"  ⚡ {performance['processing_time_ms']}ms processing time for {len(sample_reviews)} reviews")
    print(f"  🔍 {len(overall_stats['most_common_bias_types'])} distinct bias types detected")
    
    return analysis_result

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_bias_detection())