from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from docx import Document

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware (in production, you may want to restrict origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resume from DOCX file
doc = Document("20250222_Phillips_III_Ernest_Resume.docx")
resume_text = "\n".join([para.text for para in doc.paragraphs])

# Initialize the extractive question-answering pipeline using a robust model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Truncate resume to avoid exceeding the model's maximum token limit
max_context_tokens = 512  # Adjust based on your model's token limit
encoded_context = qa_pipeline.tokenizer.encode(resume_text, truncation=True, max_length=max_context_tokens)
truncated_context = qa_pipeline.tokenizer.decode(encoded_context, skip_special_tokens=True)


# Define request model for incoming JSON data
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not request.question:
        return {"error": "No question provided."}

    # Use extractive QA: the model finds the answer within the truncated resume text.
    result = qa_pipeline(question=request.question, context=truncated_context)
    answer = result.get("answer", "No answer found.")
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
