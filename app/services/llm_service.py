from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List


class LLMService:

    def __init__(self):
        # Lightweight model for local system
        model_name = "google/flan-t5-small"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_answer(self, context_chunks: List[str], question: str) -> str:
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are an AI assistant. Use the provided context to answer the question clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.2,
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
