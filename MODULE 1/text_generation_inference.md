# Simple Text Generation Inference

This guide explains how to generate text using pretrained Causal Language Models from the Hugging Face Hub.

## Theory & Concepts

### What is Text Generation?
Text generation in modern Natural Language Processing (NLP) is typically performed using **Causal Language Modeling (CLM)**. These models are trained to predict the next word (or token) in a sequence, given the previous words. By repeatedly predicting the next token and feeding it back into the model, we can generate long, coherent passages of text.

### Key Components
1.  **Tokenizer**: Machine learning models cannot process raw text directly. A tokenizer converts text into a sequence of numbers (called "tokens") that the model can understand. It also decodes the numbers back into human-readable text after generation is complete.
2.  **Causal Language Model (CLM)**: The core neural network (often a Transformer architecture) that calculates the probabilities for the next token based on the input tokens it receives.

---

## The Inference Code

Below is a simple script to perform text generation. You can easily switch the underlying model by changing the `model_name` variable.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================== CHANGE MODEL HERE ==================
model_name = "gpt2"          # Examples: "distilgpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fix for warning
tokenizer.pad_token = tokenizer.eos_token

prompt = "The future of education with AI is"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs.input_ids,
    max_length=150,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=inputs.attention_mask
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

---

## Code Explanation

Here is a breakdown of the generation parameters used in the `model.generate` function to control how text is created:

*   **`max_length=150`**: The maximum number of tokens to generate, including the length of the input prompt.
*   **`temperature=0.7`**: Controls the randomness of predictions by scaling the probabilities before picking the next token.
    *   Lower values (e.g., 0.2) make the model more deterministic and confident (safer, but potentially boring).
    *   Higher values (e.g., 0.8) make the model more creative, diverse, and random.
*   **`top_p=0.9`**: Also known as **nucleus sampling**. If set to 0.9, the model only considers the most probable tokens that add up to a 90% probability mass. This effectively filters out weird, low-probability words.
*   **`do_sample=True`**: Enables sampling algorithms (which use `temperature` and `top_p`). If set to `False`, the model uses "greedy decoding" (always picking the absolute highest probability token), which often leads to repetitive and loopy text.
*   **`pad_token_id` / `attention_mask`**: These parameters ensure the model understands where padding is, preventing generation errors and warnings in modern versions of the `transformers` library.

---

## Alternative Models to Try

You can experiment with different models by simply changing the `model_name` variable in the script. The Hugging Face `AutoModel` classes will automatically download the correct weights and architecture.

Here are some excellent open-source models for getting started, ranging from very small to slightly larger:

1.  **`gpt2`** (124M parameters)
    *   *Best for*: Learning, basic testing, and quick inference.
    *   *Pros*: Fast, lightweight, classic foundational architecture.
2.  **`distilgpt2`** (82M parameters)
    *   *Best for*: Environments with very limited memory or CPU-only inference.
    *   *Pros*: A distilled, smaller, and faster version of GPT-2.
3.  **`EleutherAI/gpt-neo-125M`** (125M parameters)
    *   *Best for*: Exploring an alternative to GPT-2.
    *   *Pros*: Developed by EleutherAI, it's a solid small model that often behaves slightly differently than OpenAI's original GPT-2.
4.  **`TinyLlama/TinyLlama-1.1B-Chat-v1.0`** (1.1B parameters)
    *   *Best for*: Chat-like interactions and higher quality text generation.
    *   *Pros*: Significantly better quality than GPT-2 and instruction-tuned for chat, yet still small enough to run on most consumer laptops.
