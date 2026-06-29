# Text Generation Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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
