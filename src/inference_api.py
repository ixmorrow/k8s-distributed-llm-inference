import logging
import sys
import time
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Force logs to be unbuffered
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
sys.stdout.reconfigure(line_buffering=True, write_through=True)

app = FastAPI()

# Load tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
logging.info("🔹 Loading model...")
generator = pipeline("text-generation", model=model_name)
logging.info("✅ Model loaded successfully!")

# Load the model normally
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
hf_model.eval()

# Compile the model for optimized inference
compiled_model = torch.compile(hf_model)


@app.get("/generate_pipeline")
async def generate_pipeline(prompt: str, max_length: int = 50):
    start_time = time.time()
    logging.info(f"🚀 Received request: {prompt}")

    result = generator(prompt, max_length=max_length, do_sample=True, truncation=True)

    latency = time.time() - start_time
    logging.info(f"⚡ Inference time: {latency:.3f} seconds")

    return {
        "generated_text": result[0]["generated_text"],
        "latency": latency,
        "method": "HuggingFacePipeline",
    }


@app.get("/generate_compiled")
async def generate_compiled(prompt: str):
    start_time = time.time()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Run inference using compiled model
    with torch.no_grad():
        outputs = compiled_model.generate(**inputs)

    # Decode the generated text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    latency = time.time() - start_time
    return {"response": output_text, "latency": latency, "method": "torch.compile"}
