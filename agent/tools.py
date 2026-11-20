import re

import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# la ruta donde se guarda la base vectorial, la uso en todas las tools
VECTOR_STORE_PATH = "./vector_store"


# chroma pide nombres de 3 a 63 caracteres, solo letras, numeros, . _ -
# y que empiecen y terminen con letra o numero
def nombre_valido(collection_name: str):
    if not collection_name or not isinstance(collection_name, str):
        return False
    return bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]", collection_name))


# saca los nombres de las colecciones, segun la version de chromadb
# a veces devuelve strings y a veces objetos
def nombres_colecciones():
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collections = client.list_collections()
    return [c if isinstance(c, str) else c.name for c in collections]


def index_data(query: str, collection_name: str):
    """
        Herramienta para indexar data en la base de datos vectorial.

    Args:
            query: str -> Texto a indexar
            collection_name: str -> Nombre de la colección a utilizar
    """
    # sin texto no hay nada que guardar
    if not query or not str(query).strip():
        return "Error: el texto a indexar esta vacio."
    # chequeo el nombre antes de que chroma tire un error feo
    if not nombre_valido(collection_name):
        return ("Error: nombre de coleccion invalido. Tiene que tener entre 3 y 63 caracteres, "
                "solo letras, numeros, puntos, guiones o guion bajo (sin espacios).")

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        docs = text_splitter.create_documents([query])

        vector_store = Chroma(
            embedding_function=OllamaEmbeddings(
                model="nomic-embed-text",
            ),
            persist_directory="./vector_store",
            collection_name=collection_name,
        )

        results = vector_store.add_documents(documents=docs)
        print("Documentos vectorizados: ", results)
        return f"Documentos vectorizados: {len(results)}"
    except Exception as e:
        return str(e)


def list_collection():
    """
        Herramienta para listar las colecciones disponibles en la base de datos vectorial.
    """
    try:
        nombres = nombres_colecciones()
        print(nombres)
        # si todavia no se guardo nada lo aviso
        if not nombres:
            return "No hay colecciones creadas todavia. Primero hay que guardar algo con index_data."
        return nombres
    except Exception as e:
        return f"Error al listar las colecciones: {e}"


def retrieval(query: str, collection_name: str, k: int = 5):
    """
        Herramienta para consultar la base de datos vectorial.

    Args:
            query: str -> consulta
            collection_name: str -> nombre de la coleccion a utilizar.
            k: int -> cantidad de documentos a obtener de la busqueda
    """
    if not query or not str(query).strip():
        return "Error: la consulta esta vacia."
    if not nombre_valido(collection_name):
        return "Error: nombre de coleccion invalido. Usa list_collection para ver las que existen."

    # por si k viene como string o cualquier cosa
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5
    if k <= 0:
        return "Error: k tiene que ser mayor a 0."
    if k > 20:
        k = 20

    try:
        # me fijo que exista, si no chroma la crea vacia y no sirve de nada
        if collection_name not in nombres_colecciones():
            return f"La coleccion '{collection_name}' no existe. Usa list_collection para ver las disponibles."

        vector_store = Chroma(
            embedding_function=OllamaEmbeddings(
                model="nomic-embed-text",
            ),
            persist_directory=VECTOR_STORE_PATH,
            collection_name=collection_name,
        )
        results = vector_store.similarity_search_with_relevance_scores(query=query, k=k)

        if not results:
            return f"No se encontro nada en '{collection_name}' para esa consulta. Proba con web_search."

        # lo paso a dict porque el agente no se lleva bien con los objetos Document
        return [
            {"contenido": doc.page_content, "score": round(score, 3)}
            for doc, score in results
        ]
    except Exception as e:
        return f"Error al consultar la base vectorial: {e}"
