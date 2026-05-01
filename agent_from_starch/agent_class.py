from rag import RAG

class Agent:
    def __init__(self, model):
        self.model = model
        self.database = RAG()

    def _generate(self,input):
        
        context = ""
