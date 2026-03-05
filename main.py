from src.llm_rag import MedicalAssistant

medi_assistant = MedicalAssistant()
result = medi_assistant.ask("Explain Acne?")
print(result)