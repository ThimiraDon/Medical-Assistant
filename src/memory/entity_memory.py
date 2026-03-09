import spacy
import re

class EntityMemory:
    """
    Tracks important medical entities mentioned in conversation,
    including user's name.
    """

    def __init__(self):

        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except:
            self.nlp = None

        self.entities = {
            "NAME": set(),      # Added for user names
            "DISEASE": set(),
            "DRUG": set(),
            "SYMPTOM": set(),
            "PROCEDURE": set()
        }

        # Patterns to detect user introductions
        self.intro_patterns = [
            r"my name is (\w+)",
            r"i am (\w+)",
            r"i'm (\w+)",
            r"this is (\w+)"
        ]

    def extract_entities(self, text: str):

        # Check for name introductions first
        for pattern in self.intro_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                name = match.group(1)
                self.entities["NAME"].add(name)

        # Then check medical entities using spacy
        if not self.nlp:
            return

        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in self.entities:
                self.entities[ent.label_].add(ent.text)

    def get_entities(self, category=None):
        """
        Returns entities as a string. Optional: filter by category
        """
        context = []

        if category:
            values = self.entities.get(category, set())
            if values:
                context.append(f"{category}: {', '.join(values)}")
        else:
            for cat, values in self.entities.items():
                if values:
                    context.append(f"{cat}: {', '.join(values)}")

        return "\n".join(context)