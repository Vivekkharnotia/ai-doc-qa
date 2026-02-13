from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# Use smaller model for low RAM systems
model_name = "google/flan-t5-small"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_answer(context: str, question: str, max_tokens: int = 128) -> str:
    """
    Generate answer using retrieved context and question.
    """

    prompt = f"""
You are an AI assistant. Use the provided context to answer the question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer.strip()
