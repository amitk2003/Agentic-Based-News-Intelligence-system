from fastapi import FastAPI, Request
from pydantic import BaseModel
from main import run_workflow  # Adapt to handle single inputs

app = FastAPI()

class NewsInput(BaseModel):
    text: str

@app.post("/process_news")
async def process_news(input: NewsInput):
    # Adapt workflow for single text input
    preprocessor = PreprocessingAgent()
    cleaned = preprocessor.clean_text(input.text)
    features = preprocessor.engineer_features([cleaned])
    
    summarizer = SummarizerAgent()
    summary = summarizer.summarize(input.text)
    
    classifier = ClassificationAgent()
    topic = classifier.classify(features)[0]
    
    decision_agent = DecisionAgent()
    action, params = decision_agent.decide(topic, summary)
    
    return {"summary": summary, "topic": topic, "decision": action, "params": params}

# Run with: uvicorn api:app --reload