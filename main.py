from agents.news_collection import NewsCollectionAgent
from agents.preprocessing import PreprocessingAgent
from agents.summarizer import SummarizerAgent
from agents.classification import ClassificationAgent
from agents.decision import DecisionAgent
from agents.action import ActionAgent

def run_workflow(api_key, query):
    collector = NewsCollectionAgent(api_key)
    raw_news = collector.collect_from_api(query)
    
    preprocessor = PreprocessingAgent()
    cleaned_news = [preprocessor.clean_text(text) for text in raw_news]
    features = preprocessor.engineer_features(cleaned_news)
    
    summarizer = SummarizerAgent()
    summaries = [summarizer.summarize(text) for text in raw_news]
    
    classifier = ClassificationAgent()
    topics = classifier.classify(features)
    
    decision_agent = DecisionAgent()
    action_agent = ActionAgent()
    
    for topic, summary in zip(topics, summaries):
        action, params = decision_agent.decide(topic, summary)
        action_agent.execute(action, params)

# Example run
run_workflow('your_newsapi_key', 'AI news')