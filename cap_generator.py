import nltk
from nltk.corpus import wordnet, brown
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from typing import List, Dict, Tuple, Optional
import random
import urllib.request
import os
import ssl

class AdvancedSpecialLetterGenerator:
    # Configuration constants
    AVG_CHUNK_SIZE = 12  # Average number of special letters per sentence chunk
    CHUNK_RANDOMNESS = 0.8  # Randomness factor for chunk size variation (±20% at 0.8)
    MAX_WORDS_PER_SENTENCE = 15  # Maximum words per sentence before splitting
    MAX_WORDS_RANDOMNESS = 5  # Randomness for max words threshold (±5)
    
    def __init__(self):
        # Fix SSL certificate verification issues on macOS
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data (run once)
        # Check and download each dataset individually
        datasets = [
            ('corpora/wordnet', 'wordnet'),
            ('tokenizers/punkt', 'punkt'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('corpora/brown', 'brown'),
            ('corpora/words', 'words')
        ]
        
        for path, name in datasets:
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading {name}...")
                nltk.download(name, quiet=True)
                print(f"Downloaded {name} successfully.")
        
        self.special_letters = set("qkfbdgjhplty")
        self.non_special_letters = set("aceimnoursuvwxz")
        
        # Load word frequency data from Brown corpus
        print("Loading word frequency data from Brown corpus...")
        self.word_freq = FreqDist(w.lower() for w in brown.words())
        print(f"Loaded frequencies for {len(self.word_freq)} words")
        
        # Load English word list - using multiple sources for reliability
        self.english_words = self._load_word_list()
        
        # Filter words to only include those that can be formed with our letter constraints
        self.valid_words = self._filter_valid_words()
        
        # Group words by their special letter patterns and POS
        self.pattern_to_words_by_pos = self._build_pattern_pos_index()

        self._setup_after_word_loading()
        
    def _load_word_list(self) -> set:
        """Load English words from multiple sources."""
        words_set = set()
        
        # Method 1: Try NLTK words corpus
        try:
            from nltk.corpus import words
            nltk.download('words', quiet=True)
            words_set.update(word.lower() for word in words.words())
            print(f"Loaded {len(words_set)} words from NLTK corpus")
        except Exception as e:
            print(f"NLTK words corpus not available: {e}")
        
        # Method 2: Extract words from WordNet (more reliable)
        try:
            all_words = set()
            for synset in wordnet.all_synsets():
                # Get lemma names (word forms)
                for lemma in synset.lemmas():
                    word = lemma.name().lower().replace('_', ' ')
                    if ' ' not in word:  # Single words only
                        all_words.add(word)
            
            words_set.update(all_words)
            print(f"Added {len(all_words)} words from WordNet, total: {len(words_set)}")
        except Exception as e:
            print(f"WordNet extraction failed: {e}")
        
        return words_set
    
    def _setup_after_word_loading(self):
        """Complete initialization after loading words."""
        # Filter words to only include those that can be formed with our letter constraints
        self.valid_words = self._filter_valid_words()
        
        # Group words by their special letter patterns and POS
        self.pattern_to_words_by_pos = self._build_pattern_pos_index()
        
        # Define recursive phrase structure grammar
        # Non-terminals can expand to other non-terminals or terminals (POS tags)
        self.phrase_structure_rules = {
            # Sentence patterns
            'S': [
                ['NP', 'VP'],              # Noun phrase + Verb phrase
                ['NP', 'VP', 'NP'],        # Subject + Verb + Object
            ],
            # Noun phrases
            'NP': [
                ['NN'],                    # Simple noun
                ['DT', 'NN'],              # Determiner + noun (the cat)
                ['JJ', 'NN'],              # Adjective + noun (big cat)
                ['DT', 'JJ', 'NN'],        # Det + Adj + noun (the big cat)
                ['DT', 'ADJP', 'NN'],      # Det + Adj phrase + noun (the very big cat)
                ['PRP'],                   # Pronoun (he, she, it)
            ],
            # Verb phrases
            'VP': [
                ['VBZ'],                   # Simple verb (runs)
                ['VBZ', 'ADVP'],           # Verb + Adverb phrase (runs fast)
                ['VBZ', 'NP'],             # Verb + Object (sees cat)
                ['VBZ', 'ADJP'],           # Verb (is) + Adjective phrase (is big)
                ['VB'],                    # Base verb (run)
                ['VB', 'NP'],              # Base verb + Object (see cat)
            ],
            # Adjective phrases
            'ADJP': [
                ['JJ'],                    # Simple adjective (big)
                ['RB', 'JJ'],              # Adverb + adjective (very big)
            ],
            # Adverb phrases
            'ADVP': [
                ['RB'],                    # Simple adverb (fast)
                ['RB', 'RB'],              # Adverb + adverb (very quickly)
            ],
        }
        
        # Generate flattened grammar rules by expanding the phrase structure
        self.grammar_rules_by_length = self._generate_grammar_rules_by_length()
        
        # Flatten for backward compatibility
        self.grammar_rules = [rule for rules in self.grammar_rules_by_length.values() for rule in rules]
    
    def _expand_phrase_structure(self, symbol: str, max_depth: int = 3, current_depth: int = 0) -> List[List[str]]:
        """Recursively expand a phrase structure symbol into all possible terminal sequences."""
        if current_depth >= max_depth:
            # At max depth, return empty to prevent infinite recursion
            return []
        
        # If it's a terminal (POS tag), return it as is
        if symbol not in self.phrase_structure_rules:
            return [[symbol]]
        
        all_expansions = []
        
        # Try each production rule for this symbol
        for production in self.phrase_structure_rules[symbol]:
            # Expand each symbol in the production
            expanded_productions = [[]]
            
            for sym in production:
                new_expansions = []
                
                if sym in self.phrase_structure_rules:
                    # Non-terminal: recursively expand it
                    sym_expansions = self._expand_phrase_structure(sym, max_depth, current_depth + 1)
                else:
                    # Terminal: use as is
                    sym_expansions = [[sym]]
                
                # Combine with previous expansions
                for prev in expanded_productions:
                    for expansion in sym_expansions:
                        new_expansions.append(prev + expansion)
                
                expanded_productions = new_expansions
            
            all_expansions.extend(expanded_productions)
        
        return all_expansions
    
    def _generate_grammar_rules_by_length(self) -> Dict[int, List[List[str]]]:
        """Generate grammar rules organized by length using phrase structure expansion."""
        print("Generating grammar rules from phrase structure...")
        
        # Expand from sentence symbol
        all_rules = self._expand_phrase_structure('S', max_depth=3)
        
        # Organize by length
        rules_by_length = {}
        for rule in all_rules:
            length = len(rule)
            if length not in rules_by_length:
                rules_by_length[length] = []
            # Avoid duplicates
            if rule not in rules_by_length[length]:
                rules_by_length[length].append(rule)
        
        # Print summary
        for length in sorted(rules_by_length.keys()):
            print(f"  Generated {len(rules_by_length[length])} rules of length {length}")
        
        return rules_by_length
    
    def _filter_valid_words(self) -> Dict[str, str]:
        """Filter words to only include those with valid letters."""
        valid_words = {}
        
        for word in self.english_words:
            word_lower = word.lower()
            # Check if word only contains our allowed letters
            if all(c in self.special_letters or c in self.non_special_letters for c in word_lower):
                # Only include words with at least one special letter
                if any(c in self.special_letters for c in word_lower):
                    valid_words[word_lower] = word_lower
        
        return valid_words
    
    def _extract_special_pattern(self, word: str) -> str:
        """Extract only the special letters from a word in order."""
        return ''.join(c for c in word.lower() if c in self.special_letters)
    
    def _get_word_pos(self, word: str) -> str:
        """Get the most common part of speech for a word using WordNet."""
        synsets = wordnet.synsets(word)
        if not synsets:
            # Fallback: use NLTK POS tagger
            tagged = pos_tag([word])
            return tagged[0][1] if tagged else 'NN'
        
        # Count POS frequencies in WordNet
        pos_counts = {}
        for synset in synsets:
            pos = synset.pos()
            # Convert WordNet POS to Penn Treebank tags
            if pos == 'n':
                pos_tag_str = 'NN'
            elif pos == 'v':
                pos_tag_str = 'VB'
            elif pos == 'a' or pos == 's':
                pos_tag_str = 'JJ'
            elif pos == 'r':
                pos_tag_str = 'RB'
            else:
                pos_tag_str = 'NN'

            pos_counts[pos_tag_str] = pos_counts.get(pos_tag_str, 0) + 1
        
        # Return most common POS
        return max(pos_counts.items(), key=lambda x: x[1])[0] if pos_counts else 'NN'
    
    def _build_pattern_pos_index(self) -> Dict[str, Dict[str, List[str]]]:
        """Build index of special patterns to words grouped by POS."""
        pattern_pos_index = {}
        
        print("Building word index with part-of-speech tags...")
        processed = 0
        total = len(self.valid_words)
        
        for word in self.valid_words:
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed}/{total} words...")
            
            pattern = self._extract_special_pattern(word)
            if pattern:  # Only include words with special letters
                pos = self._get_word_pos(word)
                
                if pattern not in pattern_pos_index:
                    pattern_pos_index[pattern] = {}
                if pos not in pattern_pos_index[pattern]:
                    pattern_pos_index[pattern][pos] = []
                
                pattern_pos_index[pattern][pos].append(word)
        
        print(f"Finished processing {total} words.")
        return pattern_pos_index
    
    def _get_words_for_pattern_and_pos(self, pattern: str, desired_pos: str) -> List[str]:
        """Get words matching a pattern and part of speech."""
        if pattern not in self.pattern_to_words_by_pos:
            return []
        
        pos_dict = self.pattern_to_words_by_pos[pattern]
        
        # Try exact match first
        if desired_pos in pos_dict:
            return pos_dict[desired_pos]
        
        # Try related POS tags
        related_pos = {
            'NN': ['NNS', 'NNP', 'NNPS'],
            'VB': ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'JJ': ['JJR', 'JJS'],
            'RB': ['RBR', 'RBS'],
        }
        
        if desired_pos in related_pos:
            for related in related_pos[desired_pos]:
                if related in pos_dict:
                    return pos_dict[related]
        
        # Return any available words for this pattern
        all_words = []
        for word_list in pos_dict.values():
            all_words.extend(word_list)
        return all_words
    
    def generate_grammatical_phrases(self, special_sequence: str, max_results: int = 10) -> List[str]:
        """Generate grammatically coherent phrases using grammar rules."""
        # Calculate expected number of words needed
        # Average: 2.0 special letters per word
        sequence_length = len(special_sequence)
        expected_words = round(sequence_length / 2.0)
        
        # Apply randomness to the max words threshold
        max_words_threshold = self.MAX_WORDS_PER_SENTENCE + random.randint(-self.MAX_WORDS_RANDOMNESS, self.MAX_WORDS_RANDOMNESS)
        
        # Average sentence: 10-15 words (20-30 special letters)
        # If sequence is too long for a single sentence, split into multiple sentences
        if expected_words > max_words_threshold:
            # Split into chunks using configured average chunk size
            chunks = self._split_into_sentence_chunks(special_sequence, avg_chunk_size=self.AVG_CHUNK_SIZE)
            
            results = []
            for _ in range(max_results):
                sentences = []
                for chunk in chunks:
                    # Generate one sentence for each chunk
                    sentence_options = self._generate_single_sentence(chunk, max_results=3)
                    if sentence_options:
                        sentences.append(random.choice(sentence_options))
                    else:
                        # If we can't generate a sentence, skip this attempt
                        break
                
                if len(sentences) == len(chunks):  # All chunks succeeded
                    result = ' '.join(sentences)
                    if result not in results:
                        results.append(result)
                
                if len(results) >= max_results:
                    break
            
            return results
        
        # Single sentence generation
        return self._generate_single_sentence(special_sequence, max_results)
    
    def _split_into_sentence_chunks(self, special_sequence: str, avg_chunk_size: int = None) -> List[str]:
        """Split a long sequence into sentence-sized chunks averaging the specified size."""
        if avg_chunk_size is None:
            avg_chunk_size = self.AVG_CHUNK_SIZE
        sequence_length = len(special_sequence)
        num_chunks = max(2, round(sequence_length / avg_chunk_size))
        
        chunks = []
        current_pos = 0
        
        for chunk_idx in range(num_chunks):
            remaining = sequence_length - current_pos
            remaining_chunks = num_chunks - chunk_idx
            
            # Target length with randomness (controlled by CHUNK_RANDOMNESS)
            target_length = remaining // remaining_chunks
            min_length = max(4, int(target_length * self.CHUNK_RANDOMNESS))
            max_length = min(remaining, int(target_length * (2 - self.CHUNK_RANDOMNESS)))
            
            if min_length > max_length:
                min_length = max_length
            
            # Randomly choose length in range
            chunk_length = random.randint(min_length, max_length)
            chunk = special_sequence[current_pos:current_pos + chunk_length]
            chunks.append(chunk)
            current_pos += chunk_length
        
        return chunks
    
    def _generate_single_sentence(self, special_sequence: str, max_results: int = 10) -> List[str]:
        """Generate a single sentence from a special sequence."""
        results = []
        
        # Calculate expected number of words needed
        sequence_length = len(special_sequence)
        # Apply slight randomness to max words limit
        max_words_limit = self.MAX_WORDS_PER_SENTENCE + random.randint(-self.MAX_WORDS_RANDOMNESS, self.MAX_WORDS_RANDOMNESS)
        expected_words = max(2, min(max_words_limit, round(sequence_length / 2.0)))
        
        # Select appropriate grammar rules based on sequence length
        selected_rules = []
        
        # First, try rules matching the expected length
        if expected_words in self.grammar_rules_by_length:
            selected_rules.extend(self.grammar_rules_by_length[expected_words])
        
        # Then try nearby lengths (±1 word)
        for offset in [1, -1, 2, -2]:
            nearby_length = expected_words + offset
            if nearby_length in self.grammar_rules_by_length:
                selected_rules.extend(self.grammar_rules_by_length[nearby_length])
        
        # Fallback to all rules if none selected
        if not selected_rules:
            selected_rules = self.grammar_rules
        
        # Try each selected grammar rule
        for rule in selected_rules:
            phrases = self._apply_grammar_rule(special_sequence, rule, max_attempts=20)
            results.extend(phrases)
            if len(results) >= max_results:
                break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for phrase in results:
            if phrase not in seen:
                seen.add(phrase)
                unique_results.append(phrase)
        
        # Capitalize first letter and add period to each sentence
        formatted_results = []
        for phrase in unique_results[:max_results]:
            if phrase:
                formatted = phrase[0].upper() + phrase[1:] + '.'
                formatted_results.append(formatted)
        
        return formatted_results
    
    def _apply_grammar_rule(self, sequence: str, rule: List[str], max_attempts: int = 20) -> List[str]:
        """Apply a specific grammar rule to generate phrases."""
        results = []
        
        for attempt in range(max_attempts):
            # Try different ways to distribute the sequence across the rule
            phrases = self._distribute_sequence_to_rule(sequence, rule)
            results.extend(phrases)
            if len(results) >= 5:  # Limit results per rule
                break
        
        return results
    
    def _distribute_sequence_to_rule(self, sequence: str, rule: List[str]) -> List[str]:
        """Distribute special letters across a grammar rule using a greedy approach."""
        if not sequence:
            return []
        
        results = []
        
        # Use a greedy approach instead of trying all combinations
        # Try a few random distributions
        for _ in range(5):  # 5 attempts per rule
            assignment = self._greedy_distribute(sequence, rule)
            if assignment:
                phrase = self._build_phrase_from_assignment(assignment, rule)
                if phrase and phrase not in results:
                    results.append(phrase)
        
        return results

    def _greedy_distribute(self, sequence: str, rule: List[str]) -> Optional[List[str]]:
        """Greedily distribute sequence across rule positions."""
        assignment = []
        remaining = sequence
        
        for i, pos in enumerate(rule):
            remaining_positions = len(rule) - i
            
            # Decide how much sequence to allocate to this position
            if remaining_positions == 1:
                # Last position gets everything remaining
                assignment.append(remaining)
            elif not remaining:
                # No letters left, use empty string (will use function word)
                assignment.append('')
            else:
                # Randomly allocate 0-3 letters to this position
                # Favor shorter allocations to leave letters for later positions
                max_alloc = min(len(remaining), 3)
                
                # Bias towards smaller values (0-2 letters per word typically)
                weights = [3, 2, 1, 1][:max_alloc + 1]
                alloc_length = random.choices(range(max_alloc + 1), weights=weights)[0]
                
                assignment.append(remaining[:alloc_length])
                remaining = remaining[alloc_length:]
        
        return assignment
    
    def _build_phrase_from_assignment(self, assignment: List[str], rule: List[str]) -> Optional[str]:
        """Build a phrase from pattern assignment and POS rule."""
        words = []
        
        for i, (pattern, pos) in enumerate(zip(assignment, rule)):
            if pattern:  # Has special letters
                word_options = self._get_words_for_pattern_and_pos(pattern, pos)
                if word_options:
                    words.append(self._weighted_word_choice(word_options))
                else:
                    return None
            else:  # No special letters, use function word
                function_word = self._get_function_word(pos)
                if function_word:
                    words.append(function_word)
                else:
                    return None
        
        return ' '.join(words) if words else None
    
    def _build_phrase_with_function_words(self, partial_assignment: List[str], 
                                         rule: List[str], start_pos: int) -> Optional[str]:
        """Complete phrase with function words for remaining positions."""
        words = []
        
        # Add words from partial assignment
        for i, pattern in enumerate(partial_assignment):
            if pattern:
                word_options = self._get_words_for_pattern_and_pos(pattern, rule[i])
                if word_options:
                    words.append(self._weighted_word_choice(word_options))
                else:
                    return None
            else:
                function_word = self._get_function_word(rule[i])
                if function_word:
                    words.append(function_word)
                else:
                    return None
        
        # Add function words for remaining positions
        for pos in rule[start_pos:]:
            function_word = self._get_function_word(pos)
            if function_word:
                words.append(function_word)
            else:
                return None
        
        return ' '.join(words) if words else None
    
    def _weighted_word_choice(self, words: List[str]) -> str:
        """Choose a word weighted by frequency, favoring common words."""
        if not words:
            return None
        
        if len(words) == 1:
            return words[0]
        
        # Get frequencies for all words (default to 1 if not in corpus)
        weights = [self.word_freq.get(word, 1) for word in words]
        
        # Apply exponential weighting to strongly favor common words
        # Square the weights to make the difference more dramatic
        weights = [w ** 1.5 for w in weights]
        
        # Use random.choices with weights (requires Python 3.6+)
        return random.choices(words, weights=weights, k=1)[0]
    
    def _get_function_word(self, pos: str) -> Optional[str]:
        """Get appropriate function words for POS tags."""
        function_words = {
            'DT': ['a', 'an', 'some'],
            'PRP': ['i', 'we'],
            'CC': ['or'],
            'IN': ['in', 'on', 'at'],
            'TO': ['to'],
        }
        
        if pos in function_words:
            return random.choice(function_words[pos])
        return None

# Example usage
def main():
    try:
        print("Initializing Advanced Special Letter Generator...")
        generator = AdvancedSpecialLetterGenerator()
        
        # Test cases
        test_cases = ["bddg", "kt", "hpy", "ldgptyldpytttt", "fg", "plt","lgtyldglyldgplylddbhttyldgpllbtyldjktylgylgplyldghbtyldpglyldgplyldgltyldpygty"]
        
        print("\nAdvanced Special Letter Phrase Generator")
        print("=" * 50)
        print("Using NLTK word corpus with part-of-speech tagging")
        print(f"Loaded {len(generator.valid_words)} valid words")
        print()
        
        for test in test_cases:
            print(f"Input: '{test}'")
            
            # Generate grammatical phrases
            print("  Grammatical phrases:")
            try:
                grammatical_phrases = generator.generate_grammatical_phrases(test, max_results=3)
                for i, phrase in enumerate(grammatical_phrases, 1):
                    print(f"    {i}. {phrase}")
                if not grammatical_phrases:
                    print("    (none found)")
            except Exception as e:
                print(f"    Error generating grammatical phrases: {e}")
                        
            print()
            
    except Exception as e:
        print(f"Error initializing generator: {e}")
        print("Make sure you have installed NLTK: pip install nltk")

if __name__ == "__main__":
    main()