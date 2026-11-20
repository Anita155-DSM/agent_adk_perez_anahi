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


# aca le explico bien el rol, que tiene que hacer y que no
instrucciones_extra = """
Ojo: web_search no es tu unica herramienta, segui el orden de abajo.

Tu objetivo es ayudar a estudiantes con dudas de programacion (python, javascript,
bases de datos, git, etc). Ademas tenes una base de conocimiento propia donde se
guardan apuntes y explicaciones.

Como trabajar:
- Si te piden guardar algo (apuntes, una explicacion, un texto), usa index_data.
  Si no te dicen en que coleccion, usa "apuntes".
- Si te preguntan algo, primero usa list_collection para ver que colecciones hay
  y despues retrieval sobre la coleccion que corresponda.
- Si retrieval no trae nada o lo que trae no alcanza para responder, recien ahi usa web_search.
- Si te preguntan que tenes guardado, usa list_collection.

Restricciones:
- Responde siempre en español, claro y sin vueltas. Si hace falta, agrega un ejemplo de codigo corto.
- No inventes datos: si las herramientas no devuelven nada util, decilo.
- Si una herramienta devuelve un error, explicale al usuario que paso y que puede hacer.
- Cuando uses info de internet, menciona de donde la sacaste (el link).
- Si la pregunta no tiene nada que ver con programacion, avisale amablemente que solo ayudas con eso.
"""

root_agent = Agent(
    model=LiteLlm("ollama_chat/qwen3:4b"),
    name='root_agent',
    description='Un asistente util que responde preguntas',
    instruction='Eres un asisistente util que responde preguntas sobre programacion. Utiliza tu herramienta llamada web_search para responder.' + instrucciones_extra,
    # agrego las tools nuevas al lado de la que ya estaba
    tools=[web_search, index_data, list_collection, retrieval],
)
