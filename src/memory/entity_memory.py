import spacy
import re


class EntityMemory:
    """
    Tracks important entities mentioned in conversation,
    including user's name and medical information.
    """

    def __init__(self):

        # Load medical spaCy model if available
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except Exception:
            self.nlp = None

        # Initialize entity storage
        self.entities = self._initialize_entities()

        # Patterns for detecting user introductions
        self.intro_patterns = [
            r"my name is ([a-zA-Z ]+)",
            r"i am ([a-zA-Z ]+)",
            r"i'm ([a-zA-Z ]+)",
            r"this is ([a-zA-Z ]+)"
        ]

    def _initialize_entities(self):
        """Creates the default entity structure."""
        return {
            "NAME": set(),
            "DISEASE": set(),
            "DRUG": set(),
            "SYMPTOM": set(),
            "PROCEDURE": set()
        }

    def extract_entities(self, text: str):
        """
        Extracts entities from user text.
        """

        text = text.strip()

        # ---- Detect name introductions ----
        for pattern in self.intro_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip().title()
                self.entities["NAME"].add(name)

        # ---- Extract medical entities using spaCy ----
        if not self.nlp:
            return

        doc = self.nlp(text)

        for ent in doc.ents:
            label = ent.label_

            if label in self.entities:
                value = ent.text.strip()
                self.entities[label].add(value)

    def get_entities(self, category=None):
        """
        Returns entities as formatted context.
        """

        context = []

        if category:
            values = self.entities.get(category, set())
            if values:
                context.append(f"{category}: {', '.join(sorted(values))}")

        else:
            for cat, values in self.entities.items():
                if values:
                    context.append(f"{cat}: {', '.join(sorted(values))}")

        return "\n".join(context)

    def clear(self):
        """
        Resets all stored entities.
        """
        self.entities = self._initialize_entities()