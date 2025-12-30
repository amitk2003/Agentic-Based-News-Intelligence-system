from transformers import pipeline

class SummarizerAgent:
    def __init__(self):
        self.summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

    def summarize(self, text):
        return self.summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']