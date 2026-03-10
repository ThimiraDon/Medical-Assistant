# src/prompts/prompt.py
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from src.prompts.fewshot_example import fewshot_examples

class MedicalPrompt:

    def __init__(self):
        self.system_prompt = SystemMessagePromptTemplate.from_template(
            """You are a professional medical assistant AI.

                Use ONLY the information provided in the context.
                    The context contains:
                    - Medical Knowledge: factual info from retrieved documents
                    - Conversation Memory: prior questions and answers

                    Rules:
                    1. Carefully read all context before answering.
                    2. Extract relevant medical information from Medical Knowledge only.
                    3. Use Conversation Memory only for continuity, not for medical facts.
                    4. If the answer is not contained in the context, respond exactly with: "I don't know."

                    Structured explanations:
                    - Use numbered/sectioned format only for detailed medical questions.
                    - Otherwise, answer concisely.

                    Output Rules:
                    - Provide clear, medically accurate explanations.
                    - Use simple, precise language.
                    - If its not a medical question answer normally. like normal Q&A.
                    - Do NOT mention sources.
                    - Do NOT fabricate information."""
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