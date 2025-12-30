import argparse
import os
import time
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = 'data/News_Category_Dataset_v3.json'
MODEL_SAVE_PATH = 'models/classifier.pkl'
VECTORIZER_SAVE_PATH = 'models/vectorizer.pkl'


# =======================
# ✅ CLASSIFICATION AGENT
# =======================
class ClassificationAgent:
    def __init__(self,
                 model_path=MODEL_SAVE_PATH,
                 vectorizer_path=VECTORIZER_SAVE_PATH):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            self.model = None
            self.vectorizer = None

    def predict(self, cleaned_texts: list[str]):
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not trained. Run training first.")

        features = self.vectorizer.transform(cleaned_texts)
        preds = self.model.predict(features)
        probs = self.model.predict_proba(features).max(axis=1)

        return [
            {"label": label, "confidence": float(conf)}
            for label, conf in zip(preds, probs)
        ]


# =======================
# TRAINING PIPELINE
# =======================
def load_data(sample: int | None = None) -> pd.DataFrame:
    print("Loading dataset...")
    df = pd.read_json(DATA_PATH, lines=True)
    if sample is not None:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
    print(f"Dataset shape: {df.shape}")
    return df


def get_preprocessor():
    try:
        from .preprocessing import PreprocessingAgent
        return PreprocessingAgent
    except Exception:
        from preprocessing import PreprocessingAgent
        return PreprocessingAgent


def train(sample: int | None = None, max_features: int = 10000):
    os.makedirs('models', exist_ok=True)

    PreprocessingAgent = get_preprocessor()
    df = load_data(sample)

    texts = (df['headline'].fillna('') + ' ' +
             df['short_description'].fillna('')).str.strip()
    labels = df['category']

    preprocessor = PreprocessingAgent()
    cleaned_texts = [preprocessor.clean_text(t) for t in texts]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )
    features = vectorizer.fit_transform(cleaned_texts)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = LogisticRegression(
        max_iter=1000,
        solver='saga',
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)

    print("Model and vectorizer saved.")


# =======================
# CLI ENTRY POINT
# =======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int)
    parser.add_argument('--max-features', type=int, default=10000)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    if args.fast and args.sample is None:
        args.sample = 2000
        args.max_features = 1000

    train(args.sample, args.max_features)
