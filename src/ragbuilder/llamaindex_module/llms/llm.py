from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI    
def getLLM(**kwargs):
    if kwargs['retreival_model'] == "openai":
        return OpenAI(temperature=0.2, model="gpt-4")
    elif kwargs['retreival_model'] == "mistralai":
        return MistralAI(model="mistral-small")
    