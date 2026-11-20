import os

from ddgs import DDGS
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

# las otras tools las tengo en tools.py
from .tools import index_data, list_collection, retrieval

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"


def web_search(query: str, max_results: int = 5):
    """
    Herramienta para busquedas en internet utilizando
    duckduckgo_search

    Args:
        query: str -> La consulta buscar en internet.
        max_results: int -> cantidad maxima de resultados.
    """
    try:
        results = DDGS().text(query=query,
                              max_results=max_results)
        return results
    except Exception as e:
        print("Error interno al utilizar la herramienta", e)
        return str(e)


root_agent = Agent(
    model=LiteLlm("ollama_chat/qwen3:4b"),
    name='root_agent',
    description='Un asistente util que responde preguntas',
    instruction='Eres un asisistente util que responde preguntas sobre programacion. Utiliza tu herramienta llamada web_search para responder.',
    # agrego las tools nuevas al lado de la que ya estaba
    tools=[web_search, index_data, list_collection, retrieval],
)
