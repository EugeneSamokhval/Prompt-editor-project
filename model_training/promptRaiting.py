import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class PromptRatingHandler:
    MODEL_WEIGHTS = {
        "gpt":  {"clarity": 0.30, "descriptive": 0.25, "context": 0.20, "style": 0.05, "composition": 0.0, "lighting": 0.0, "technical": 0.05, "negative": 0.0},
        "stable_diffusion": {"clarity": 0.30, "descriptive": 0.20, "context": 0.10, "style": 0.15, "composition": 0.10, "lighting": 0.05, "technical": 0.05, "negative": 0.20}}

    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def calculate(prompt: str):
        pos_tagged_prompt = nltk.pos_tag(prompt)
        words_count = len(pos_tagged_prompt)
