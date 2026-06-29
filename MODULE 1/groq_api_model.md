# Groq API Model

```python
from groq import Groq

client = Groq(api_key="your-groq-api-key-here")

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "system", "content": "You are a helpful teaching assistant."},
        {"role": "user", "content": "Explain Generative AI in simple terms for students."}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
```
