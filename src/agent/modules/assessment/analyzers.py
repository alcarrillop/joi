"""
Analyzers for automatic assessment of vocabulary, grammar and fluency
"""
import re
from typing import List, Dict, Set, Tuple
from datetime import datetime

from .models import (
    VocabularyAnalysis, GrammarAnalysis, FluencyAnalysis, 
    LanguageSkill, ErrorSeverity
)

# spaCy opcional
nlp = None
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    print("Info: spaCy not available. Using simplified text analysis.")
    nlp = None

class VocabularyAnalyzer:
    """Vocabulary analyzer based on CEFR levels"""
    
    def __init__(self):
        # Vocabulary by CEFR level (representative samples)
        self.vocabulary_levels = {
            "A1": {
                "hello", "goodbye", "yes", "no", "please", "thank", "you", "I", "am", "is", "are",
                "name", "age", "from", "live", "like", "have", "go", "come", "eat", "drink",
                "good", "bad", "big", "small", "hot", "cold", "happy", "sad", "mother", "father",
                "family", "friend", "home", "work", "food", "water", "time", "day", "night"
            },
            "A2": {
                "yesterday", "tomorrow", "weekend", "holiday", "travel", "hotel", "restaurant",
                "delicious", "expensive", "cheap", "beautiful", "interesting", "boring", "tired",
                "excited", "worried", "comfortable", "difficult", "easy", "important", "necessary",
                "probably", "usually", "sometimes", "always", "never", "often", "already", "yet"
            },
            "B1": {
                "experience", "opportunity", "responsibility", "environment", "government", "society",
                "technology", "definitely", "absolutely", "particularly", "especially", "recently",
                "immediately", "eventually", "unfortunately", "obviously", "apparently", "basically",
                "generally", "personally", "seriously", "actually", "currently", "previously"
            },
            "B2": {
                "consequently", "furthermore", "nevertheless", "nonetheless", "meanwhile", "therefore",
                "significantly", "substantially", "increasingly", "particularly", "specifically",
                "dramatically", "considerably", "approximately", "undoubtedly", "extraordinary",
                "sophisticated", "comprehensive", "fundamental", "substantial", "controversial"
            }
        }
        
        # Function words that don't count for level
        self.function_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "by", "for", "with",
            "to", "of", "it", "he", "she", "they", "we", "this", "that", "these", "those"
        }
    
    def analyze_vocabulary(self, text: str) -> VocabularyAnalysis:
        """Analyze vocabulary used in the text"""
        # Clean and tokenize
        words = self._extract_words(text)
        content_words = [w for w in words if w.lower() not in self.function_words]
        
        total_words = len(words)
        unique_words = len(set(words))
        
        # Classify words by level
        basic_words = []
        advanced_words = []
        level_scores = {"A1": 0, "A2": 0, "B1": 0, "B2": 0}
        
        for word in content_words:
            word_lower = word.lower()
            word_level = self._get_word_level(word_lower)
            
            if word_level in ["A1", "A2"]:
                basic_words.append(word)
            elif word_level in ["B1", "B2"]:
                advanced_words.append(word)
            
            # Only count words of known levels
            if word_level in level_scores:
                level_scores[word_level] += 1
        
        # Determine general vocabulary level
        vocabulary_level = self._determine_vocabulary_level(level_scores, len(content_words))
        
        # Calculate scores
        complexity_score = self._calculate_complexity_score(level_scores, len(content_words))
        appropriateness_score = self._calculate_appropriateness_score(words, vocabulary_level)
        
        return VocabularyAnalysis(
            words_used=words,
            total_words=total_words,
            unique_words=unique_words,
            advanced_words=advanced_words,
            basic_words=basic_words,
            vocabulary_level=vocabulary_level,
            complexity_score=complexity_score,
            appropriateness_score=appropriateness_score
        )
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text"""
        # Clean text and extract words
        cleaned = re.sub(r'[^\w\s]', '', text)
        words = cleaned.split()
        return [w for w in words if w.strip()]
    
    def _get_word_level(self, word: str) -> str:
        """Determine the CEFR level of a word"""
        for level, vocab_set in self.vocabulary_levels.items():
            if word in vocab_set:
                return level
        return "UNKNOWN"
    
    def _determine_vocabulary_level(self, level_scores: Dict[str, int], total_content_words: int) -> str:
        """Determine general level based on words used"""
        if total_content_words == 0:
            return "A1"
        
        # Calculate percentages
        percentages = {level: count / total_content_words for level, count in level_scores.items()}
        
        # Logic to determine level
        if percentages["B2"] > 0.3:
            return "B2"
        elif percentages["B1"] > 0.4:
            return "B1"
        elif percentages["A2"] > 0.5:
            return "A2"
        else:
            return "A1"
    
    def _calculate_complexity_score(self, level_scores: Dict[str, int], total_words: int) -> float:
        """Calculate vocabulary complexity score"""
        if total_words == 0:
            return 0.0
        
        weights = {"A1": 0.1, "A2": 0.3, "B1": 0.6, "B2": 1.0}
        weighted_score = sum(level_scores[level] * weights[level] for level in level_scores)
        return min(weighted_score / total_words, 1.0)
    
    def _calculate_appropriateness_score(self, words: List[str], level: str) -> float:
        """Calculate how appropriate the vocabulary is for the context"""
        # Simplified implementation - can be improved with context analysis
        if len(words) < 3:
            return 0.5
        
        # For now, assume using vocabulary at the appropriate level is good
        level_weights = {"A1": 0.6, "A2": 0.7, "B1": 0.8, "B2": 0.9}
        return level_weights.get(level, 0.5)

class GrammarAnalyzer:
    """Grammar analyzer using regex patterns and linguistic rules"""
    
    def __init__(self):
        self.grammar_patterns = {
            "subject_verb_agreement": [
                (r'\b(I|you|we|they)\s+(is)\b', "Use 'are' with I/you/we/they"),
                (r'\b(he|she|it)\s+(are)\b', "Use 'is' with he/she/it"),
            ],
            "verb_tense": [
                (r'\byesterday\s+.*\b(go|come|see)\b', "Use past tense with 'yesterday'"),
                (r'\btomorrow\s+.*\b(went|came|saw)\b', "Use future tense with 'tomorrow'"),
            ],
            "article_usage": [
                (r'\ba\s+[aeiou]', "Use 'an' before vowel sounds"),
                (r'\ban\s+[bcdfghjklmnpqrstvwxyz]', "Use 'a' before consonant sounds"),
            ]
        }
        
        # Patterns to detect verb tenses
        self.tense_patterns = {
            "present": [r'\b(am|is|are)\b', r'\b\w+s\b(?=\s|$)', r'\b(have|has)\b'],
            "past": [r'\b(was|were)\b', r'\b\w+ed\b', r'\b(went|came|saw|did|had)\b'],
            "future": [r'\bwill\b', r'\bgoing to\b', r'\bshall\b'],
            "continuous": [r'\b(am|is|are|was|were)\s+\w+ing\b', r'\bhave been\s+\w+ing\b']
        }
    
    def analyze_grammar(self, text: str) -> GrammarAnalysis:
        """Analyze text grammar"""
        errors_detected = []
        sentence_structures = []
        tenses_used = []
        
        # Detect errors with patterns
        for error_type, patterns in self.grammar_patterns.items():
            for pattern, explanation in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    errors_detected.append({
                        "type": error_type,
                        "original": match.group(),
                        "position": match.span(),
                        "explanation": explanation
                    })
        
        # Detect verb tenses using patterns
        tenses_used = self._extract_tenses_simple(text)
        
        # Analyze sentence structures in simplified way
        sentence_structures = self._analyze_sentence_structures_simple(text)
        
        # Calculate grammar level
        grammar_level = self._determine_grammar_level(errors_detected, sentence_structures, text)
        
        # Calculate scores
        accuracy_score = self._calculate_accuracy_score(errors_detected, text)
        complexity_score = self._calculate_grammar_complexity(sentence_structures, tenses_used)
        
        return GrammarAnalysis(
            errors_detected=errors_detected,
            error_count=len(errors_detected),
            sentence_structures=sentence_structures,
            tenses_used=tenses_used,
            grammar_level=grammar_level,
            accuracy_score=accuracy_score,
            complexity_score=complexity_score
        )
    
    def _extract_tenses_simple(self, text: str) -> List[str]:
        """Extract verb tenses using regex patterns"""
        tenses = set()
        text_lower = text.lower()
        
        for tense, patterns in self.tense_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tenses.add(tense)
        
        return list(tenses)
    
    def _analyze_sentence_structures_simple(self, text: str) -> List[str]:
        """Analyze sentence structures in simplified way"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        structures = []
        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)
            
            # Simplified analysis based on length and conjunctions
            if word_count <= 5:
                structures.append("basic")
            elif any(conj in sentence.lower() for conj in ["and", "but", "or", "because", "although"]):
                structures.append("compound")
            elif word_count > 15:
                structures.append("complex")
            else:
                structures.append("simple")
        
        return structures
    
    def _determine_grammar_level(self, errors: List[Dict], structures: List[str], text: str) -> str:
        """Determine grammar level"""
        error_rate = len(errors) / max(len(text.split()), 1)
        
        # Structural complexity analysis
        complex_structures = sum(1 for s in structures if s in ["complex", "compound"])
        structure_complexity = complex_structures / max(len(structures), 1)
        
        if error_rate < 0.1 and structure_complexity > 0.5:
            return "B2"
        elif error_rate < 0.2 and structure_complexity > 0.3:
            return "B1"
        elif error_rate < 0.3:
            return "A2"
        else:
            return "A1"
    
    def _calculate_accuracy_score(self, errors: List[Dict], text: str) -> float:
        """Calculate grammar accuracy score"""
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        error_rate = len(errors) / word_count
        return max(0.0, 1.0 - error_rate * 2)  # Penalize errors
    
    def _calculate_grammar_complexity(self, structures: List[str], tenses: List[str]) -> float:
        """Calculate grammar complexity"""
        if not structures:
            return 0.0
        
        # Score based on variety of structures and tenses
        structure_variety = len(set(structures)) / 4  # maximum 4 types
        tense_variety = len(set(tenses)) / 4  # maximum 4 main tenses
        
        return min((structure_variety + tense_variety) / 2, 1.0)

class FluencyAnalyzer:
    """Fluency analyzer for conversation"""
    
    def analyze_fluency(self, text: str, response_time: float = 0.0) -> FluencyAnalysis:
        """Analyze text fluency"""
        message_length = len(text.split())
        
        # Calculate fluency scores
        coherence_score = self._analyze_coherence(text)
        clarity_score = self._analyze_clarity(text)
        natural_flow_score = self._analyze_natural_flow(text, response_time)
        
        return FluencyAnalysis(
            response_time=response_time,
            message_length=message_length,
            coherence_score=coherence_score,
            clarity_score=clarity_score,
            natural_flow_score=natural_flow_score
        )
    
    def _analyze_coherence(self, text: str) -> float:
        """Analyze text coherence"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 0.8  # Short message, assume coherence
        
        # Simplified analysis: look for connectors and references
        connectors = ["and", "but", "because", "so", "then", "also", "however", "therefore"]
        connector_count = sum(1 for word in text.lower().split() if word in connectors)
        
        # Score based on connector usage
        coherence = min(0.5 + (connector_count / len(sentences)) * 0.5, 1.0)
        return coherence
    
    def _analyze_clarity(self, text: str) -> float:
        """Analyze message clarity"""
        words = text.split()
        if not words:
            return 0.0
        
        # Clarity factors
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_sentence_length = len(words) / max(sentence_count, 1)
        
        # Score based on appropriate length
        length_score = 1.0 - abs(avg_word_length - 5) / 10  # ideal words ~5 letters
        sentence_score = 1.0 - abs(avg_sentence_length - 10) / 20  # ideal sentences ~10 words
        
        return max(0.0, (length_score + sentence_score) / 2)
    
    def _analyze_natural_flow(self, text: str, response_time: float) -> float:
        """Analyze natural conversation flow"""
        words = text.split()
        word_count = len(words)
        
        # Response time analysis
        if response_time > 0:
            # Ideal time per word (approx. 1-2 seconds)
            ideal_time = word_count * 1.5
            time_score = 1.0 - abs(response_time - ideal_time) / ideal_time
            time_score = max(0.0, min(1.0, time_score))
        else:
            time_score = 0.7  # Neutral if no time
        
        # Text naturalness analysis
        # Look for excessive repetitions
        word_freq = {}
        for word in words:
            word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
        
        repetition_penalty = sum(1 for freq in word_freq.values() if freq > 2) / len(words)
        repetition_score = max(0.0, 1.0 - repetition_penalty)
        
        return (time_score + repetition_score) / 2 