import nltk
import math
from prompts_tokenizer import PromptTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from enum import Enum

class InputTypes(Enum):
    GPT = 'gpt'
    SD = 'sd'
    FALLBACK = 'fallback'


class PromptRatingHandler:
    MAX_WORDS_WEIGHTED = {
        InputTypes.GPT: {
            "clarity": 3600,
            "descriptive": 3000,
            "context": 2400,
            "style": 600,
            "composition": 0,
            "lighting": 0,
            "technical": 600,
            "negative": 0,
        },
        InputTypes.SD: {
            "clarity": 450,
            "descriptive": 300,
            "context": 150,
            "style": 225,
            "composition": 150,
            "lighting": 75,
            "technical": 75,
            "negative": 300,
        },
        InputTypes.FALLBACK: {
            "clarity": 50,
            "descriptive": 250,
            "context": 25,
            "style": 25,
            "composition": 50,
            "lighting": 25,
            "technical": 25,
            "negative": 50,
        },
    }

    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.prompt_tokenizer = PromptTokenizer()

    def calculate(self, prompt: str, ai_type: InputTypes = InputTypes.FALLBACK):
        print(prompt)
        weighted_max_words = self.MAX_WORDS_WEIGHTED.get(ai_type)
        number_of_tokens = len(word_tokenize(prompt))
        tokenized_prompt = self.prompt_tokenizer.tokenize_prompt(prompt)
        rating = 0
        all_max_tokens = sum(weighted_max_words.values())
        for key in weighted_max_words.keys():
            number_of_current_type_entries = len(tokenized_prompt.get(key))
            if (number_of_current_type_entries /number_of_tokens) <= (weighted_max_words.get(key) /all_max_tokens ):
                rating+=(number_of_current_type_entries /number_of_tokens)
            else:
                rating+= (2 * weighted_max_words.get(key) /all_max_tokens ) - (number_of_current_type_entries /number_of_tokens)
        rating = rating * ( 2 * number_of_tokens/ all_max_tokens)
        print(rating)
        return math.ceil((rating) * 1000)


