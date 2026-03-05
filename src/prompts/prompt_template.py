# src/prompts/prompt.py
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from src.prompts.fewshot_example import fewshot_examples

class MedicalPrompt:

    def __init__(self):
        self.system_prompt = SystemMessagePromptTemplate.from_template(
            "You are a medical assistant AI. Answer accurately using the provided context. "
            "If the answer is not in the context, respond: 'I don't know.' "
            "Always follow the structured format in examples."
        )

        # Convert modular few-shot examples to AIMessagePromptTemplates
        self.ai_examples = [
            AIMessagePromptTemplate.from_template(
                f"Example:\nQuestion: {ex['question']}\nAnswer:\n{ex['answer']}"
            )
            for ex in fewshot_examples
        ]

        # User query template
        self.user_prompt = HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\nQuestion:\n{input}"
            )

        # Combine into ChatPromptTemplate
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_prompt, *self.ai_examples, self.user_prompt]
        )

    def get_prompt(self):
        return self.chat_prompt