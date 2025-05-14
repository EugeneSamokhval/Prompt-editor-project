import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string


class PromptTokenizer:

    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        self.wn_lemmas = set(wordnet.all_lemma_names())

        self.KEYWORDS = {
            "clarity":      {"clarity", "clear", "clearly", "clearness"},
            "descriptive":  {"describe", "describes", "described", "describing",
                             "description", "descriptions", "descriptive"},
            "context":      {"context", "contexts", "contextual"},
            "style":        {"style", "styles", "stylistic"},
            "composition":  {"composition", "compositions", "compositional",
                             "compose", "composed", "composing"},
            "lighting":     {"lighting", "light", "lights", "lit", "lighted"},
            "technical":    {"technical", "technically"},
            "negative":     {"negative", "negatives", "negativity", "negatively"},
        }

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
        lemmatizer = WordNetLemmatizer()
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
            lemma = lemmatizer.lemmatize(word, pos)
            if lemma not in self.wn_lemmas:
                continue
            synset = wordnet.synset(lemma + '.' + pos + '.01')
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
            for category, keyword_set in self.KEYWORDS.items():
                for keyword in keyword_set:
                    key_synset = wordnet.synset(
                        keyword + '.' + pos + '.01')
                    inclusion_dict[category] += key_synset.lch_similarity(
                        synset)
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
        tokenized = nltk.pos_tag(tokenized)
        tokenized = self._get_wordnet_pos_tags(tokenized)
        result = self._token_processing(tokenized)
        return result


prompt = " anime aesthetics, view from the center of the hollow tree, big ancient dragon sitting in the center, 4 wings, stone scales, dim light, dark souls aesteticss, epic view, ,atmospheric perspective, perspective, tall shot, UHD, masterpiece, accurate, super detail, high details, high quality, award winning, 16k"
tokenizer = PromptTokenizer()
print(tokenizer.tokenize_prompt(prompt))
