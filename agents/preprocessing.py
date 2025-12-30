import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class PreprocessingAgent:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Remove symbols
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)

    def engineer_features(self, texts):
        cleaned = [self.clean_text(text) for text in texts]
        return self.vectorizer.fit_transform(cleaned)  # Returns TF-IDF matrix