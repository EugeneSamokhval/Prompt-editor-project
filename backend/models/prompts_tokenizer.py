import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
import copy

EMPTY_PROMPT_PARTS = {
    "clarity": [],
    "descriptive": [],
    "context": [],
    "style": [],
    "composition": [],
    "lighting": [],
    "technical": [],
    "negative": []
}


class PromptTokenizer:

    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('universal_tagset')
        self.wn_lemmas = set(wordnet.all_lemma_names())
        self.lemmatizer = WordNetLemmatizer()
        KEYWORDS = {
            'clarity': [
                'sharp', 'crisp', 'defined', 'detailed', 'precise', 'distinct', 'clean', 'harsh', 'hifi', 'highres',
                'resolution', 'refined', 'articulate', 'legible', 'unblurred', 'unobscured', 'transparent', 'lucid',
                'readable', 'explicit', 'intelligible', 'accurate', 'pinpoint', 'micro', 'crystalline'
            ],
            'descriptive': [
                'depict', 'illustrate', 'portray', 'render', 'characterize', 'narrate', 'express', 'chronicle', 'elaborate',
                'specify', 'outline', 'delineate', 'sketch', 'represent', 'evoke', 'convey', 'articulate', 'summarize',
                'interpret', 'visualize', 'recount', 'paint', 'relate', 'define', 'exemplify'
            ],
            'context': [
                'setting', 'environment', 'backdrop', 'surroundings', 'scenario', 'milieu', 'atmosphere', 'ambience',
                'background', 'situation', 'framework', 'circumstance', 'location', 'place', 'contextual', 'mood',
                'vibe', 'tone', 'theme', 'story', 'narrative', 'world', 'space', 'arena', 'stage'
            ],
            'style': [
                'aesthetic', 'stylized', 'artistic', 'visual', 'look', 'feel', 'design', 'vibe', 'mood', 'tone',
                'flair', 'expression', 'genre', 'school', 'movement', 'trend', 'fashion', 'signature', 'identity',
                'character', 'texture', 'form', 'pattern', 'motif', 'composition'
            ],
            'composition': [
                'arrangement', 'balance', 'framing', 'layout', 'placement', 'structure', 'organization', 'geometry',
                'symmetry', 'asymmetry', 'ruleofthirds', 'leadinglines', 'negativepace', 'depth', 'layering',
                'foreground', 'midground', 'background', 'focus', 'flow', 'harmony', 'contrast', 'alignment',
                'proportion', 'grid'
            ],
            'lighting': [
                'illumination', 'brightness', 'glow', 'shade', 'shadow', 'highlight', 'contrast', 'exposure',
                'backlight', 'rimlight', 'ambient', 'directional', 'diffuse', 'harsh', 'soft', 'warm', 'cool',
                'natural', 'artificial', 'dappled', 'spotlight', 'moody', 'dramatic', 'chiaroscuro', 'lowkey'
            ],
            'technical': [
                'accurate', 'precise', 'detailed', 'intricate', 'highres', '4k', '8k', 'hdr', 'uhd', 'rendering',
                'specification', 'calibrated', 'optimized', 'engineered', 'perfected', 'refined', 'polished',
                'flawless', 'professional', 'mastered', 'authentic', 'realistic', 'scientific', 'measured', 'exact'
            ],
            'negative': [
                'blurry', 'grainy', 'noisy', 'distorted', 'pixelated', 'overexposed', 'underexposed', 'washedout',
                'muddy', 'hazy', 'foggy', 'unfocused', 'chaotic', 'cluttered', 'disorganized', 'unbalanced',
                'jarring', 'unpleasant', 'unappealing', 'dull', 'flat', 'lifeless', 'boring', 'unrefined', 'amateur'
            ]
        }
        self.lemmatized_keywords = copy.deepcopy(EMPTY_PROMPT_PARTS)
        for keyword, keyword_items in KEYWORDS.items():
            lemmatized_keyword_items = [
                self.lemmatizer.lemmatize(word) for word in keyword_items]
            self.lemmatized_keywords[keyword] = [tagged_entry for tagged_entry in nltk.pos_tag(
                lemmatized_keyword_items) if (tagged_entry not in self.lemmatized_keywords[keyword])]

        self.stop_words = set(stopwords.words("english"))
        for sign in string.punctuation:
            if sign not in self.stop_words:
                self.stop_words.add(sign)

    def _nltk_to_wordnet(self, pos_tag: str):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _get_wordnet_pos_tags(self, pairs_list: list[tuple[str, str]]):
        result = []
        for pair in pairs_list:
            result.append((pair[0], self._nltk_to_wordnet(pair[1])))
        return result

    def _token_processing(self, pairs_list: list[tuple[str, str]]):

        results = {
            "clarity": [],
            "descriptive": [],
            "context": [],
            "style": [],
            "composition": [],
            "lighting": [],
            "technical": [],
            "negative": []
        }
        for (word, pos) in pairs_list:
            lemma = self.lemmatizer.lemmatize(word, pos)
            if lemma not in self.wn_lemmas:
                continue
            synsets = wordnet.synsets(lemma, pos)

            if len(synsets) == 0:
                continue

            inclusion_dict = {
                "clarity": 0,
                "descriptive": 0,
                "context": 0,
                "style": 0,
                "composition": 0,
                "lighting": 0,
                "technical": 0,
                "negative": 0
            }
            for category, keyword_set in self.lemmatized_keywords.items():
                pos_tagged_keyword_set = self._get_wordnet_pos_tags(
                    keyword_set)

                for keyword, key_pos in pos_tagged_keyword_set:
                    key_synset = wordnet.synsets(
                        keyword, key_pos)
                    if (pos == key_pos) and (len(key_synset) > 0):
                        inclusion_dict[category] = max(key_synset[0].lch_similarity(
                            synsets[0]), inclusion_dict[category])

            max_weight = max(inclusion_dict.values())
            for save_category, value in inclusion_dict.items():
                if (value == max_weight):
                    results[save_category].append(word)
                    break

        return results

    def tokenize_prompt(self, prompt: str):
        tokenized = word_tokenize(prompt)
        tokenized = [
            word.lower() for word in tokenized if word not in self.stop_words]
        tokenized = nltk.pos_tag(tokenized, tagset="universal")
        tokenized = self._get_wordnet_pos_tags(tokenized)
        result = self._token_processing(tokenized)
        return result
