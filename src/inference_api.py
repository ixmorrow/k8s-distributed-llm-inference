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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
logging.info("🔹 Loading model...")
generator = pipeline("text-generation", model=model_name)
logging.info("✅ Model loaded successfully!")

# Load model and move to device (GPU is available)
hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    hf_model, {torch.nn.Linear}, dtype=torch.qint8
).to(device)
quantized_model.eval()

# Compile the model for optimized inference
hf_model = torch.compile(quantized_model)
logging.info("✅ Quantized and compiled model with PyTorch!")
logging.info(f"✅ Model loaded on {device}")


@app.on_event("startup")
async def warmup():
    dummy_prompt = "Hello, world!"
    inputs = tokenizer(dummy_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        hf_model.generate(**inputs)
    logging.info("🔥 Model warmed up!")


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
        "device": str(device),
    }


@app.get("/generate_compiled")
async def generate_compiled(prompt: str, max_length: int = 50):
    start_time = time.time()
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Run inference using compiled model
    with torch.no_grad():
        outputs = hf_model.generate(**inputs, max_length=max_length, do_sample=True)

    # Decode the generated text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    latency = time.time() - start_time

    return {
        "response": output_text,
        "latency": latency,
        "method": "torch.compile",
        "device": str(device),
    }
