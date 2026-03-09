import spacy

class EntityMemory:
    """
    Tracks important medical entities mentioned in conversation.
    """


    def __init__(self):

        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except:
            self.nlp = None

        self.entities = {
            "DISEASE": set(),
            "DRUG": set(),
            "SYMPTOM": set(),
            "PROCEDURE": set()
        }

    def extract_entities(self, text: str):

        if not self.nlp:
            return

        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ in self.entities:
                self.entities[ent.label_].add(ent.text)

    def get_entities(self):

        context = []

        for category, values in self.entities.items():
            if values:
                context.append(f"{category}: {', '.join(values)}")

        return "\n".join(context)

