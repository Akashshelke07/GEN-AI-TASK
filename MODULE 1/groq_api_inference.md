# Groq API Inference Guide

This guide explains how to use the Groq API to run powerful open-source Large Language Models (LLMs).

## Why Use Groq?

Groq provides a highly optimized, cloud-based inference engine powered by specialized hardware (LPUs - Language Processing Units). 
We use Groq for educational and lab purposes primarily because:

1.  **Generous Free Tier**: Groq offers a completely free tier that is more than sufficient for learning, building simple apps, and classroom environments. No credit card is required to sign up.
2.  **Blazing Fast Inference**: Groq's specialized hardware makes it one of the fastest ways to generate text with LLMs, significantly improving the user experience and reducing waiting time during development.
3.  **Access to Open-Source Models**: It provides access to high-quality, open-weights models like Meta's LLaMA 3 and Mistral's Mixtral, allowing you to experiment with state-of-the-art AI without expensive subscriptions.

---

## How to Get Started

### Step 1: Get Your Free API Key
1.  Go to the Groq Console: [https://console.groq.com/keys](https://console.groq.com/keys)
2.  Sign in using your Google or GitHub account.
3.  Click on "Create API Key" and copy the generated key. **Keep this key secret!**

### Step 2: The Inference Code

Below is a simple script to generate text using the Groq API. Make sure to replace `"your-groq-api-key-here"` with your actual key.

```python
from groq import Groq

# Initialize the Groq client with your API key
client = Groq(api_key="your-groq-api-key-here")

# Make an API call to generate a response
response = client.chat.completions.create(
    model="llama3-8b-8192",        # Free & Fast model
    messages=[
        {"role": "system", "content": "You are a helpful teaching assistant."},
        {"role": "user", "content": "Explain Generative AI in simple terms for students."}
    ],
    max_tokens=200,
    temperature=0.7
)

# Print the generated text
print(response.choices[0].message.content)
```

---

## Code Explanation

Here is a breakdown of the key parameters in the API call:

*   **`model="llama3-8b-8192"`**: Specifies which AI model to use. `llama3-8b-8192` is an excellent, fast model for general tasks. The `8192` refers to its context window (how much text it can remember in one go).
*   **`messages`**: This is a list of dictionaries representing the conversation history.
    *   **`"role": "system"`**: Used to set the behavior, persona, or rules for the AI (e.g., "You are a helpful teaching assistant.").
    *   **`"role": "user"`**: This contains the actual prompt or question you want the AI to answer.
*   **`max_tokens=200`**: The maximum number of tokens (roughly parts of words) the model is allowed to generate in its response. This helps control costs (if on a paid tier) and prevents overly long, rambling answers.
*   **`temperature=0.7`**: Controls the creativity or randomness of the response.
    *   `0.0` - `0.3`: More focused, deterministic, and factual.
    *   `0.7` - `1.0`: More creative, diverse, and unpredictable.

---

## Alternative Free Models on Groq

You can easily swap out the `model` parameter in the code to try different open-source AI models available on Groq's free tier:

*   **`"llama3-8b-8192"`**: (Recommended) Extremely fast and highly capable for most standard tasks and learning.
*   **`"llama3-70b-8192"`**: A much larger, more powerful version of LLaMA 3. It's smarter and better at complex reasoning, though slightly slower to generate text compared to the 8B version.
*   **`"mixtral-8x7b-32768"`**: A powerful "Mixture of Experts" model by Mistral AI. It's a great alternative to LLaMA models and features a much larger context window (32,768 tokens), making it better for tasks involving large amounts of text.
