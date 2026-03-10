class EpisodicMemory:
    """
    Stores important conversation events.
    """

    def __init__(self, max_events=20):

        self.max_events = max_events
        self.events = []

    def add_event(self, event: str):

        self.events.append(event)

        if len(self.events) > self.max_events:
            self.events.pop(0)

    def get_events(self):

        return "\n".join(self.events)
    
    def clear(self):
        self.events = []

