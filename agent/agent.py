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
    # si la query viene vacia no tiene sentido buscar
    if not query or not str(query).strip():
        return "Error: la consulta esta vacia, pasame algo para buscar."

    # por si el modelo manda un numero raro o un string
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 5
    if max_results <= 0:
        return "Error: max_results tiene que ser mayor a 0."
    # le pongo un tope asi no trae una banda de resultados
    if max_results > 10:
        max_results = 10

    try:
        results = DDGS().text(query=query,
                              max_results=max_results)
        # si no encontro nada aviso, asi el modelo no inventa
        if not results:
            return f"No se encontraron resultados en internet para: '{query}'."
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
