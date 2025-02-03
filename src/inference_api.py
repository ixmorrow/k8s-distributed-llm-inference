from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()  # Ensure this is correctly named

# Load model
generator = pipeline("text-generation", model="gpt2")


@app.get("/generate")
def generate_text(prompt: str, max_length: int = 50):
    result = generator(prompt, max_length=max_length, do_sample=True)
    return {"generated_text": result[0]["generated_text"]}
