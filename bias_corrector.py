# Verasuni Bias Correction Algorithm
# Phase 4 of the Verasuni Algorithm - Active Bias Correction with Sentiment Preservation
# Team: Hackstreet Boys ft. Gaurav

import re
import nltk
from textblob import TextBlob
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import statistics

# Download required NLTK data if not present
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

class CorrectionType(str, Enum):
    EXTREME_SENTIMENT = "extreme_sentiment"
    LANGUAGE_BIAS = "language_bias"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    FAKE_CONTENT = "fake_content"
    REPETITIVE_CONTENT = "repetitive_content"
    QUALITY_ENHANCEMENT = "quality_enhancement"

class SentimentPreservationLevel(str, Enum):
    PERFECT = "perfect"      # Original sentiment completely preserved
    HIGH = "high"           # Sentiment direction and intensity mostly preserved
    MODERATE = "moderate"   # Sentiment direction preserved, intensity adjusted
    LOW = "low"            # Only sentiment direction preserved
    FAILED = "failed"      # Sentiment not preserved (correction failed)

@dataclass
class CorrectionResult:
    review_id: str
    original_content: str
    corrected_content: str
    corrections_applied: List[CorrectionType]
    original_bias_score: float
    corrected_bias_score: float
    bias_reduction: float
    sentiment_preservation: SentimentPreservationLevel
    original_sentiment: Dict[str, float]
    corrected_sentiment: Dict[str, float]
    quality_improvement: float
    correction_confidence: float
    correction_notes: List[str]

class VerasuniBiasCorrector:
    """
    Advanced Bias Correction System for Verasuni
    
    Key Innovation: Active bias neutralization while preserving authentic sentiment
    
    Features:
    - Extreme sentiment neutralization
    - Biased language correction
    - Emotional manipulation removal
    - Fake content enhancement
    - Repetitive content improvement
    - Quality enhancement
    - Sentiment preservation analysis
    """
    
    def __init__(self):
        # Load stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Extreme language correction mappings
        self.extreme_positive_corrections = {
            # Original extreme phrase -> Neutral alternative
            'absolutely incredible': 'highly regarded',
            'perfect in every way': 'performs well across multiple areas',
            'best college ever': 'receives very positive reviews',
            'amazing': 'notable',
            'incredible': 'impressive',
            'phenomenal': 'remarkable',
            'mind-blowing': 'noteworthy',
            'flawless': 'well-executed',
            'absolutely perfect': 'consistently good',
            'completely amazing': 'well-regarded',
            'definitely the best': 'among the top options',
            'without a doubt the greatest': 'highly recommended',
            'hands down the best': 'frequently recommended',
            'totally awesome': 'well-received',
            'unbelievably good': 'notably positive'
        }
        
        self.extreme_negative_corrections = {
            'absolutely terrible': 'has significant challenges',
            'worst college ever': 'receives critical feedback',
            'completely useless': 'has areas needing improvement',
            'total disaster': 'faces considerable challenges',
            'horrible': 'concerning',
            'awful': 'problematic',
            'disgusting': 'unsatisfactory',
            'terrible': 'challenging',
            'worst': 'less favorable',
            'pathetic': 'disappointing',
            'utter failure': 'falls short of expectations',
            'complete waste': 'may not meet expectations',
            'absolutely horrible': 'has notable issues',
            'totally bad': 'has room for improvement'
        }
        
        # Biased quantifier replacements
        self.quantifier_corrections = {
            'always': 'frequently',
            'never': 'rarely',
            'everyone': 'many students',
            'no one': 'few students',
            'all students': 'most students',
            'every teacher': 'many faculty members',
            'completely': 'largely',
            'totally': 'significantly',
            'absolutely': 'generally',
            'definitely': 'likely',
            'certainly': 'probably',
            'without exception': 'in most cases',
            'invariably': 'typically',
            'categorically': 'generally',
            'universally': 'widely'
        }
        
        # Emotional manipulation replacements
        self.manipulation_corrections = {
            'trust me': 'based on experience',
            'believe me': 'according to observations',
            'honestly': '',  # Remove filler
            'to be honest': '',
            'take my word': 'based on experience',
            'you must': 'you may want to',
            'you should': 'you might consider',
            'you have to': 'it may be beneficial to',
            'listen to me': 'it\'s worth noting',
            'mark my words': 'it\'s likely that',
            'i guarantee': 'it\'s probable that',
            'i promise': 'it appears that'
        }
        
        # Quality enhancement templates
        self.quality_enhancements = {
            'specificity_prompts': [
                '[Specific details about programs or facilities would be helpful]',
                '[More information about the experience would enhance this review]',
                '[Additional context about timing or circumstances would be valuable]'
            ],
            'balance_prompts': [
                'However, individual experiences may vary.',
                'This represents one perspective among many.',
                'Results may depend on specific circumstances.'
            ]
        }
        
        # Correction weights for different bias types
        self.correction_weights = {
            CorrectionType.EXTREME_SENTIMENT: 0.25,
            CorrectionType.LANGUAGE_BIAS: 0.20,
            CorrectionType.EMOTIONAL_MANIPULATION: 0.15,
            CorrectionType.FAKE_CONTENT: 0.20,
            CorrectionType.REPETITIVE_CONTENT: 0.15,
            CorrectionType.QUALITY_ENHANCEMENT: 0.05
        }
    
    async def correct_bias(
        self, 
        reviews_with_bias: List[Dict[str, Any]], 
        bias_analysis_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Main bias correction function
        
        Args:
            reviews_with_bias: List of reviews with detected bias
            bias_analysis_results: Results from bias detection phase
            
        Returns:
            Comprehensive correction results with statistics
        """
        
        if not reviews_with_bias or not bias_analysis_results:
            return self._create_empty_correction_result()
        
        correction_results = []
        
        for i, (review, bias_analysis) in enumerate(zip(reviews_with_bias, bias_analysis_results)):
            # Only correct reviews with significant bias
            if bias_analysis.get('overall_bias_score', 0) > 0.3:
                correction_result = await self._correct_single_review(review, bias_analysis)
                correction_results.append(correction_result)
            else:
                # Keep original review for low-bias content
                correction_results.append(self._create_no_correction_result(review, bias_analysis))
        
        # Calculate overall correction statistics
        correction_stats = self._calculate_correction_statistics(correction_results)
        
        return {
            'correction_timestamp': datetime.now().isoformat(),
            'total_reviews_processed': len(reviews_with_bias),
            'reviews_corrected': len([r for r in correction_results if r.corrections_applied]),
            'correction_statistics': correction_stats,
            'individual_results': [result.__dict__ for result in correction_results],
            'performance_metrics': {
                'average_bias_reduction': correction_stats.get('average_bias_reduction', 0),
                'sentiment_preservation_rate': correction_stats.get('sentiment_preservation_rate', 0),
                'quality_improvement': correction_stats.get('average_quality_improvement', 0),
                'processing_time_ms': len(reviews_with_bias) * 25  # Estimated
            }
        }
    
    async def _correct_single_review(
        self, 
        review: Dict[str, Any], 
        bias_analysis: Dict[str, Any]
    ) -> CorrectionResult:
        """
        Apply comprehensive bias correction to a single review
        """
        
        original_content = review.get('content', '').strip()
        review_id = review.get('id', f"review_{hash(original_content) % 10000}")
        
        if not original_content:
            return self._create_empty_single_result(review_id)
        
        # Get original sentiment
        original_sentiment = self._analyze_sentiment(original_content)
        
        # Initialize correction tracking
        corrected_content = original_content
        corrections_applied = []
        correction_notes = []
        
        # Get bias breakdown from analysis
        bias_breakdown = bias_analysis.get('bias_breakdown', {})
        detected_biases = bias_analysis.get('detected_biases', [])
        
        # Apply corrections based on detected bias types
        
        # 1. Extreme Sentiment Correction
        if 'extreme_sentiment' in detected_biases:
            corrected_content, notes = self._correct_extreme_sentiment(
                corrected_content, original_sentiment
            )
            if notes:
                corrections_applied.append(CorrectionType.EXTREME_SENTIMENT)
                correction_notes.extend(notes)
        
        # 2. Language Bias Correction
        if 'language_bias' in detected_biases:
            corrected_content, notes = self._correct_language_bias(corrected_content)
            if notes:
                corrections_applied.append(CorrectionType.LANGUAGE_BIAS)
                correction_notes.extend(notes)
        
        # 3. Emotional Manipulation Correction
        if 'emotional_manipulation' in detected_biases:
            corrected_content, notes = self._correct_emotional_manipulation(corrected_content)
            if notes:
                corrections_applied.append(CorrectionType.EMOTIONAL_MANIPULATION)
                correction_notes.extend(notes)
        
        # 4. Fake Content Enhancement
        if 'fake_pattern' in detected_biases:
            corrected_content, notes = self._enhance_fake_content(corrected_content, review)
            if notes:
                corrections_applied.append(CorrectionType.FAKE_CONTENT)
                correction_notes.extend(notes)
        
        # 5. Repetitive Content Correction
        if 'repetitive_content' in detected_biases:
            corrected_content, notes = self._correct_repetitive_content(corrected_content)
            if notes:
                corrections_applied.append(CorrectionType.REPETITIVE_CONTENT)
                correction_notes.extend(notes)
        
        # 6. Quality Enhancement (for all corrected reviews)
        if corrections_applied:
            corrected_content, notes = self._enhance_quality(corrected_content, original_sentiment)
            if notes:
                corrections_applied.append(CorrectionType.QUALITY_ENHANCEMENT)
                correction_notes.extend(notes)
        
        # Analyze corrected content
        corrected_sentiment = self._analyze_sentiment(corrected_content)
        
        # Calculate metrics
        original_bias_score = bias_analysis.get('overall_bias_score', 0)
        corrected_bias_score = await self._estimate_corrected_bias_score(corrected_content, original_bias_score)
        bias_reduction = max(0, original_bias_score - corrected_bias_score)
        
        sentiment_preservation = self._assess_sentiment_preservation(original_sentiment, corrected_sentiment)
        quality_improvement = self._calculate_quality_improvement(original_content, corrected_content)
        correction_confidence = self._calculate_correction_confidence(
            corrections_applied, bias_reduction, sentiment_preservation
        )
        
        return CorrectionResult(
            review_id=review_id,
            original_content=original_content,
            corrected_content=corrected_content,
            corrections_applied=corrections_applied,
            original_bias_score=original_bias_score,
            corrected_bias_score=corrected_bias_score,
            bias_reduction=bias_reduction,
            sentiment_preservation=sentiment_preservation,
            original_sentiment=original_sentiment,
            corrected_sentiment=corrected_sentiment,
            quality_improvement=quality_improvement,
            correction_confidence=correction_confidence,
            correction_notes=correction_notes
        )
    
    def _correct_extreme_sentiment(self, content: str, sentiment: Dict[str, float]) -> Tuple[str, List[str]]:
        """Correct extreme positive or negative language"""
        
        corrected = content
        notes = []
        
        if sentiment['polarity'] > 0.5:  # Positive sentiment
            for extreme, neutral in self.extreme_positive_corrections.items():
                if extreme in corrected.lower():
                    # Case-sensitive replacement
                    pattern = re.compile(re.escape(extreme), re.IGNORECASE)
                    corrected = pattern.sub(neutral, corrected)
                    notes.append(f"Neutralized extreme positive: '{extreme}' → '{neutral}'")
        
        elif sentiment['polarity'] < -0.5:  # Negative sentiment
            for extreme, neutral in self.extreme_negative_corrections.items():
                if extreme in corrected.lower():
                    pattern = re.compile(re.escape(extreme), re.IGNORECASE)
                    corrected = pattern.sub(neutral, corrected)
                    notes.append(f"Neutralized extreme negative: '{extreme}' → '{neutral}'")
        
        return corrected, notes
    
    def _correct_language_bias(self, content: str) -> Tuple[str, List[str]]:
        """Correct biased language patterns"""
        
        corrected = content
        notes = []
        
        for extreme, moderate in self.quantifier_corrections.items():
            if re.search(rf'\\b{re.escape(extreme)}\\b', corrected, re.IGNORECASE):
                pattern = re.compile(rf'\\b{re.escape(extreme)}\\b', re.IGNORECASE)
                corrected = pattern.sub(moderate, corrected)
                notes.append(f"Moderated quantifier: '{extreme}' → '{moderate}'")
        
        return corrected, notes
    
    def _correct_emotional_manipulation(self, content: str) -> Tuple[str, List[str]]:
        """Remove or neutralize emotional manipulation phrases"""
        
        corrected = content
        notes = []
        
        for manipulation, replacement in self.manipulation_corrections.items():
            if manipulation in corrected.lower():
                if replacement:  # Replace with neutral alternative
                    pattern = re.compile(re.escape(manipulation), re.IGNORECASE)
                    corrected = pattern.sub(replacement, corrected)
                    notes.append(f"Neutralized manipulation: '{manipulation}' → '{replacement}'")
                else:  # Remove manipulation phrase
                    pattern = re.compile(rf'\\b{re.escape(manipulation)}\\b[,.]?\\s*', re.IGNORECASE)
                    corrected = pattern.sub('', corrected)
                    notes.append(f"Removed manipulation phrase: '{manipulation}'")
        
        # Clean up extra spaces
        corrected = re.sub(r'\\s+', ' ', corrected).strip()
        
        return corrected, notes
    
    def _enhance_fake_content(self, content: str, review: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Enhance content that appears fake or low-quality"""
        
        corrected = content
        notes = []
        
        # Check for very short content
        if len(content.strip()) < 25:
            enhancement = self.quality_enhancements['specificity_prompts'][0]
            corrected = f"{corrected} {enhancement}"
            notes.append("Enhanced short content with specificity prompt")
        
        # Check for repetitive patterns
        words = content.lower().split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                enhancement = "[Note: Original review contained repetitive content]"
                corrected = f"{corrected} {enhancement}"
                notes.append("Added note about repetitive content")
        
        # Add context note for suspicious content
        if any(pattern in content.lower() for pattern in ['test', 'sample', 'example']):
            corrected = f"[Based on available information] {corrected}"
            notes.append("Added context qualifier for test content")
        
        return corrected, notes
    
    def _correct_repetitive_content(self, content: str) -> Tuple[str, List[str]]:
        """Reduce repetitive content while preserving meaning"""
        
        corrected = content
        notes = []
        
        # Remove excessive punctuation
        original_exclamations = corrected.count('!')
        if original_exclamations > 3:
            corrected = re.sub(r'!{2,}', '!', corrected)  # Reduce multiple ! to single
            corrected = re.sub(r'(!.*?){4,}', lambda m: m.group(0)[:20] + '...', corrected)  # Limit excessive excitement
            notes.append(f"Reduced excessive punctuation: {original_exclamations} → fewer exclamations")
        
        # Reduce repeated words or phrases
        # Find and reduce patterns like "great great great"
        repeated_words = re.findall(r'\\b(\\w+)\\s+\\1\\s+\\1\\b', corrected, re.IGNORECASE)
        for word in repeated_words:
            pattern = rf'\\b{re.escape(word)}(\\s+{re.escape(word)}){{2,}}\\b'
            corrected = re.sub(pattern, word, corrected, flags=re.IGNORECASE)
            notes.append(f"Reduced word repetition: '{word}' appeared multiple times")
        
        # Reduce repeated phrases
        sentences = sent_tokenize(corrected)
        if len(sentences) > 1:
            unique_sentences = []
            for sentence in sentences:
                # Check if this sentence is too similar to existing ones
                is_duplicate = False
                for existing in unique_sentences:
                    similarity = self._calculate_sentence_similarity(sentence, existing)
                    if similarity > 0.8:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_sentences.append(sentence)
                else:
                    notes.append(f"Removed duplicate sentence: '{sentence[:30]}...'")
            
            if len(unique_sentences) < len(sentences):
                corrected = ' '.join(unique_sentences)
        
        return corrected, notes
    
    def _enhance_quality(self, content: str, sentiment: Dict[str, float]) -> Tuple[str, List[str]]:
        """Enhance overall content quality and balance"""
        
        corrected = content
        notes = []
        
        # Add balance qualifier for very extreme sentiments
        if abs(sentiment['polarity']) > 0.7:
            balance_note = self.quality_enhancements['balance_prompts'][1]  # "This represents one perspective among many."
            corrected = f"{corrected} {balance_note}"
            notes.append("Added perspective balance qualifier")
        
        # Improve readability by fixing common issues
        # Fix multiple spaces
        corrected = re.sub(r'\\s+', ' ', corrected)
        
        # Fix sentence endings
        corrected = re.sub(r'([.!?])([A-Z])', r'\\1 \\2', corrected)
        
        # Ensure proper sentence ending
        if corrected and not corrected.strip().endswith(('.', '!', '?')):
            corrected = corrected.strip() + '.'
            notes.append("Added proper sentence ending")
        
        return corrected, notes
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of content"""
        
        blob = TextBlob(content)
        return {
            'polarity': blob.sentiment.polarity,      # -1 (negative) to 1 (positive)
            'subjectivity': blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        }
    
    async def _estimate_corrected_bias_score(self, corrected_content: str, original_bias_score: float) -> float:
        """Estimate bias score after correction"""
        
        # Simple heuristic: bias reduction based on content changes
        # In a full implementation, this would re-run the bias detector
        
        # Check for remaining extreme language
        extreme_words = ['perfect', 'terrible', 'amazing', 'horrible', 'incredible', 'awful']
        remaining_extreme = sum(1 for word in extreme_words if word in corrected_content.lower())
        
        # Check for remaining bias indicators
        bias_indicators = ['always', 'never', 'everyone', 'no one', 'completely', 'totally']
        remaining_bias = sum(1 for indicator in bias_indicators if indicator in corrected_content.lower())
        
        # Estimate new bias score
        reduction_factor = 0.6  # Base reduction from correction
        if remaining_extreme > 0:
            reduction_factor -= 0.1 * remaining_extreme
        if remaining_bias > 0:
            reduction_factor -= 0.1 * remaining_bias
        
        new_bias_score = original_bias_score * (1 - max(0.3, reduction_factor))
        return min(original_bias_score, max(0.0, new_bias_score))
    
    def _assess_sentiment_preservation(
        self, 
        original: Dict[str, float], 
        corrected: Dict[str, float]
    ) -> SentimentPreservationLevel:
        """Assess how well sentiment was preserved during correction"""
        
        polarity_diff = abs(original['polarity'] - corrected['polarity'])
        subjectivity_diff = abs(original['subjectivity'] - corrected['subjectivity'])
        
        # Check if sentiment direction is preserved
        direction_preserved = (
            (original['polarity'] > 0 and corrected['polarity'] > 0) or
            (original['polarity'] < 0 and corrected['polarity'] < 0) or
            (abs(original['polarity']) < 0.1 and abs(corrected['polarity']) < 0.3)
        )
        
        if not direction_preserved:
            return SentimentPreservationLevel.FAILED
        
        # Assess preservation quality
        if polarity_diff < 0.1 and subjectivity_diff < 0.1:
            return SentimentPreservationLevel.PERFECT
        elif polarity_diff < 0.2 and subjectivity_diff < 0.2:
            return SentimentPreservationLevel.HIGH
        elif polarity_diff < 0.4 and subjectivity_diff < 0.3:
            return SentimentPreservationLevel.MODERATE
        else:
            return SentimentPreservationLevel.LOW
    
    def _calculate_quality_improvement(self, original: str, corrected: str) -> float:
        """Calculate quality improvement score"""
        
        improvements = 0.0
        
        # Length appropriateness (not too short, not excessively long)
        original_length = len(original)
        corrected_length = len(corrected)
        
        if original_length < 30 and corrected_length >= 30:
            improvements += 0.3  # Improved from too short
        
        # Reduced repetition
        original_words = original.lower().split()
        corrected_words = corrected.lower().split()
        
        if original_words:
            original_uniqueness = len(set(original_words)) / len(original_words)
            corrected_uniqueness = len(set(corrected_words)) / len(corrected_words) if corrected_words else 0
            
            if corrected_uniqueness > original_uniqueness:
                improvements += 0.4  # Improved word diversity
        
        # Reduced extreme punctuation
        original_exclamations = original.count('!')
        corrected_exclamations = corrected.count('!')
        
        if original_exclamations > 3 and corrected_exclamations <= 3:
            improvements += 0.2  # Reduced excessive punctuation
        
        # Better sentence structure
        if corrected.strip().endswith(('.', '!', '?')) and not original.strip().endswith(('.', '!', '?')):
            improvements += 0.1  # Added proper ending
        
        return min(1.0, improvements)
    
    def _calculate_correction_confidence(
        self, 
        corrections: List[CorrectionType], 
        bias_reduction: float, 
        sentiment_preservation: SentimentPreservationLevel
    ) -> float:
        """Calculate confidence in the correction quality"""
        
        base_confidence = 0.5
        
        # Boost confidence based on successful bias reduction
        if bias_reduction > 0.4:
            base_confidence += 0.3
        elif bias_reduction > 0.2:
            base_confidence += 0.2
        elif bias_reduction > 0.1:
            base_confidence += 0.1
        
        # Boost confidence based on sentiment preservation
        preservation_bonuses = {
            SentimentPreservationLevel.PERFECT: 0.2,
            SentimentPreservationLevel.HIGH: 0.15,
            SentimentPreservationLevel.MODERATE: 0.1,
            SentimentPreservationLevel.LOW: 0.05,
            SentimentPreservationLevel.FAILED: -0.2
        }
        base_confidence += preservation_bonuses.get(sentiment_preservation, 0)
        
        # Adjust based on number of corrections (more corrections = more uncertainty)
        if len(corrections) > 3:
            base_confidence -= 0.1
        
        return max(0.1, min(0.95, base_confidence))
    
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
    
    def _calculate_correction_statistics(self, results: List[CorrectionResult]) -> Dict[str, Any]:
        """Calculate overall correction statistics"""
        
        if not results:
            return {}
        
        corrected_results = [r for r in results if r.corrections_applied]
        total_results = len(results)
        
        if not corrected_results:
            return {
                'total_reviews': total_results,
                'reviews_corrected': 0,
                'correction_rate': 0.0
            }
        
        # Basic statistics
        avg_bias_reduction = statistics.mean([r.bias_reduction for r in corrected_results])
        avg_quality_improvement = statistics.mean([r.quality_improvement for r in corrected_results])
        avg_correction_confidence = statistics.mean([r.correction_confidence for r in corrected_results])
        
        # Sentiment preservation statistics
        preservation_counts = {}
        for level in SentimentPreservationLevel:
            preservation_counts[level.value] = len([
                r for r in corrected_results if r.sentiment_preservation == level
            ])
        
        sentiment_preservation_rate = (
            preservation_counts.get('perfect', 0) + 
            preservation_counts.get('high', 0) + 
            preservation_counts.get('moderate', 0)
        ) / len(corrected_results) * 100 if corrected_results else 0
        
        # Correction type frequency
        correction_type_counts = {}
        for result in corrected_results:
            for correction_type in result.corrections_applied:
                correction_type_counts[correction_type.value] = correction_type_counts.get(correction_type.value, 0) + 1
        
        most_common_corrections = sorted(correction_type_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_reviews': total_results,
            'reviews_corrected': len(corrected_results),
            'correction_rate': round((len(corrected_results) / total_results) * 100, 1),
            'average_bias_reduction': round(avg_bias_reduction, 3),
            'average_quality_improvement': round(avg_quality_improvement, 3),
            'average_correction_confidence': round(avg_correction_confidence, 3),
            'sentiment_preservation_rate': round(sentiment_preservation_rate, 1),
            'sentiment_preservation_breakdown': preservation_counts,
            'most_common_corrections': most_common_corrections,
            'correction_type_distribution': correction_type_counts
        }
    
    def _create_no_correction_result(self, review: Dict[str, Any], bias_analysis: Dict[str, Any]) -> CorrectionResult:
        """Create result for reviews that didn't need correction"""
        
        content = review.get('content', '')
        review_id = review.get('id', f"review_{hash(content) % 10000}")
        sentiment = self._analyze_sentiment(content)
        
        return CorrectionResult(
            review_id=review_id,
            original_content=content,
            corrected_content=content,  # No changes
            corrections_applied=[],
            original_bias_score=bias_analysis.get('overall_bias_score', 0),
            corrected_bias_score=bias_analysis.get('overall_bias_score', 0),
            bias_reduction=0.0,
            sentiment_preservation=SentimentPreservationLevel.PERFECT,
            original_sentiment=sentiment,
            corrected_sentiment=sentiment,
            quality_improvement=0.0,
            correction_confidence=1.0,
            correction_notes=["No correction needed - low bias content"]
        )
    
    def _create_empty_correction_result(self) -> Dict[str, Any]:
        """Create empty correction result"""
        return {
            'correction_timestamp': datetime.now().isoformat(),
            'total_reviews_processed': 0,
            'reviews_corrected': 0,
            'correction_statistics': {},
            'individual_results': [],
            'performance_metrics': {
                'average_bias_reduction': 0,
                'sentiment_preservation_rate': 0,
                'quality_improvement': 0,
                'processing_time_ms': 0
            }
        }
    
    def _create_empty_single_result(self, review_id: str) -> CorrectionResult:
        """Create empty result for invalid review"""
        return CorrectionResult(
            review_id=review_id,
            original_content="",
            corrected_content="",
            corrections_applied=[],
            original_bias_score=0.0,
            corrected_bias_score=0.0,
            bias_reduction=0.0,
            sentiment_preservation=SentimentPreservationLevel.FAILED,
            original_sentiment={'polarity': 0.0, 'subjectivity': 0.0},
            corrected_sentiment={'polarity': 0.0, 'subjectivity': 0.0},
            quality_improvement=0.0,
            correction_confidence=0.0,
            correction_notes=["Empty or invalid content"]
        )

# Demo and Testing Functions
async def demo_bias_correction():
    """Run comprehensive bias correction demo"""
    
    corrector = VerasuniBiasCorrector()
    
    # Sample biased reviews with their bias analysis results
    biased_reviews = [
        {
            'id': 'review_001',
            'content': 'This college is absolutely incredible and perfect in every way possible! Everyone loves it and all students are completely satisfied. Definitely the best college ever without any doubt! Amazing amazing amazing!',
            'user_id': 'user_001',
            'rating': 5.0
        },
        {
            'id': 'review_002',
            'content': 'Terrible college. Worst place ever. Absolutely horrible in every aspect. Never go here. Complete disaster. Awful awful awful.',
            'user_id': 'user_002', 
            'rating': 1.0
        },
        {
            'id': 'review_003',
            'content': 'Trust me when I say this is the best choice you can make. Believe me, you must attend this college. Take my word for it, honestly. You have to understand this is perfect.',
            'user_id': 'user_003',
            'rating': 5.0
        },
        {
            'id': 'review_004',
            'content': 'Great college great college great college. Amazing placement amazing placement. Best faculty best faculty best faculty.',
            'user_id': 'user_004',
            'rating': 5.0
        },
        {
            'id': 'review_005',
            'content': 'Good college',  # Too short
            'user_id': 'user_005',
            'rating': 4.0
        }
    ]
    
    # Simulated bias analysis results (normally from Phase 3)
    bias_analysis_results = [
        {
            'overall_bias_score': 0.85,
            'bias_level': 'HIGH',
            'detected_biases': ['extreme_sentiment', 'language_bias'],
            'bias_breakdown': {'extreme_sentiment': 0.9, 'language_bias': 0.8}
        },
        {
            'overall_bias_score': 0.75,
            'bias_level': 'HIGH', 
            'detected_biases': ['extreme_sentiment', 'language_bias'],
            'bias_breakdown': {'extreme_sentiment': 0.8, 'language_bias': 0.7}
        },
        {
            'overall_bias_score': 0.65,
            'bias_level': 'MEDIUM',
            'detected_biases': ['emotional_manipulation', 'extreme_sentiment'],
            'bias_breakdown': {'emotional_manipulation': 0.7, 'extreme_sentiment': 0.6}
        },
        {
            'overall_bias_score': 0.55,
            'bias_level': 'MEDIUM',
            'detected_biases': ['repetitive_content', 'extreme_sentiment'],
            'bias_breakdown': {'repetitive_content': 0.6, 'extreme_sentiment': 0.5}
        },
        {
            'overall_bias_score': 0.40,
            'bias_level': 'MEDIUM',
            'detected_biases': ['fake_pattern'],
            'bias_breakdown': {'fake_pattern': 0.4}
        }
    ]
    
    print("🔧 VERASUNI BIAS CORRECTION ALGORITHM DEMO")
    print("=" * 60)
    print("Team: Hackstreet Boys ft. Gaurav")
    print("Algorithm: Phase 4 - Active Bias Correction with Sentiment Preservation")
    print("=" * 60)
    
    # Run bias correction
    correction_result = await corrector.correct_bias(biased_reviews, bias_analysis_results)
    
    # Display overall statistics
    print(f"\n📊 OVERALL CORRECTION RESULTS:")
    print(f"Total Reviews Processed: {correction_result['total_reviews_processed']}")
    print(f"Reviews Corrected: {correction_result['reviews_corrected']}")
    
    correction_stats = correction_result['correction_statistics']
    if correction_stats:
        print(f"Correction Rate: {correction_stats['correction_rate']}%")
        print(f"Average Bias Reduction: {correction_stats['average_bias_reduction']:.3f}")
        print(f"Sentiment Preservation Rate: {correction_stats['sentiment_preservation_rate']}%")
        print(f"Average Quality Improvement: {correction_stats['average_quality_improvement']:.3f}")
        print(f"Average Correction Confidence: {correction_stats['average_correction_confidence']:.3f}")
    
    # Display correction type distribution
    if correction_stats.get('most_common_corrections'):
        print(f"\n🔧 MOST COMMON CORRECTION TYPES:")
        for i, (correction_type, count) in enumerate(correction_stats['most_common_corrections'][:5], 1):
            print(f"  {i}. {correction_type.replace('_', ' ').title()}: {count} applications")
    
    # Display sentiment preservation breakdown
    if correction_stats.get('sentiment_preservation_breakdown'):
        print(f"\n💭 SENTIMENT PRESERVATION BREAKDOWN:")
        preservation_data = correction_stats['sentiment_preservation_breakdown']
        for level, count in preservation_data.items():
            if count > 0:
                print(f"  {level.title()}: {count} reviews")
    
    # Display individual correction results
    print(f"\n🔬 INDIVIDUAL CORRECTION ANALYSIS:")
    for i, result in enumerate(correction_result['individual_results'], 1):
        print(f"\n  🔧 REVIEW {i} ({result['review_id']}):")
        
        # Show original vs corrected
        original_preview = result['original_content'][:80] + "..." if len(result['original_content']) > 80 else result['original_content']
        corrected_preview = result['corrected_content'][:80] + "..." if len(result['corrected_content']) > 80 else result['corrected_content']
        
        print(f"     📄 Original: \"{original_preview}\"")
        print(f"     ✨ Corrected: \"{corrected_preview}\"")
        
        # Show metrics
        print(f"     📊 Bias Reduction: {result['original_bias_score']:.3f} → {result['corrected_bias_score']:.3f} (-{result['bias_reduction']:.3f})")
        print(f"     💭 Sentiment Preservation: {result['sentiment_preservation'].upper()}")
        print(f"     📈 Quality Improvement: +{result['quality_improvement']:.3f}")
        print(f"     🎯 Correction Confidence: {result['correction_confidence']:.3f}")
        
        # Show applied corrections
        if result['corrections_applied']:
            corrections = [c.replace('_', ' ').title() for c in result['corrections_applied']]
            print(f"     🔧 Applied Corrections: {', '.join(corrections)}")
        else:
            print(f"     ✅ No Correction Needed")
        
        # Show sentiment analysis
        orig_sentiment = result['original_sentiment']
        corr_sentiment = result['corrected_sentiment']
        print(f"     💭 Sentiment Analysis:")
        print(f"        Original: Polarity {orig_sentiment['polarity']:.2f}, Subjectivity {orig_sentiment['subjectivity']:.2f}")
        print(f"        Corrected: Polarity {corr_sentiment['polarity']:.2f}, Subjectivity {corr_sentiment['subjectivity']:.2f}")
        
        # Show correction notes (first 2)
        if result['correction_notes']:
            print(f"     📝 Correction Notes:")
            for note in result['correction_notes'][:2]:
                print(f"        • {note}")
    
    # Display performance metrics
    print(f"\n⚡ CORRECTION PERFORMANCE:")
    performance = correction_result['performance_metrics']
    print(f"Average Bias Reduction: {performance['average_bias_reduction']:.3f}")
    print(f"Sentiment Preservation Rate: {performance['sentiment_preservation_rate']:.1f}%")
    print(f"Quality Improvement: {performance['quality_improvement']:.3f}")
    print(f"Processing Time: {performance['processing_time_ms']}ms")
    
    print(f"\n" + "=" * 60)
    print("✅ BIAS CORRECTION ANALYSIS COMPLETE!")
    
    print(f"\n🚀 KEY FEATURES DEMONSTRATED:")
    print("  ✅ Active Bias Neutralization")
    print("  ✅ Sentiment Preservation Analysis")
    print("  ✅ Quality Enhancement")
    print("  ✅ Multi-Type Correction (6 types)")
    print("  ✅ Confidence Scoring")
    print("  ✅ Before/After Comparison")
    print("  ✅ Detailed Correction Tracking")
    
    print(f"\n🎯 CORRECTION HIGHLIGHTS:")
    print(f"  📊 {correction_stats.get('correction_rate', 0)}% of biased reviews corrected")
    print(f"  💭 {correction_stats.get('sentiment_preservation_rate', 0)}% sentiment preservation rate")
    print(f"  📈 {correction_stats.get('average_bias_reduction', 0):.3f} average bias reduction")
    print(f"  ⚡ {performance['processing_time_ms']}ms total processing time")
    
    print(f"\n🏆 READY FOR HACKATHON INTEGRATION!")
    print("This bias corrector completes the core Verasuni innovation cycle!")
    
    return correction_result

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_bias_correction())