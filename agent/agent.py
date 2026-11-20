import os

from ddgs import DDGS
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

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
    tools=[web_search],
)
