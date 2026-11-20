import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# la ruta donde se guarda la base vectorial, la uso en todas las tools
VECTOR_STORE_PATH = "./vector_store"


def index_data(query: str, collection_name: str):
    """
        Herramienta para indexar data en la base de datos vectorial.

    Args:
            query: str -> Texto a indexar
            collection_name: str -> Nombre de la colección a utilizar
    """
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
        client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        collections = client.list_collections()
        # segun la version de chromadb a veces devuelve strings y a veces objetos
        nombres = [c if isinstance(c, str) else c.name for c in collections]
        print(nombres)
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
    try:
        vector_store = Chroma(
            embedding_function=OllamaEmbeddings(
                model="nomic-embed-text",
            ),
            persist_directory=VECTOR_STORE_PATH,
            collection_name=collection_name,
        )
        results = vector_store.similarity_search_with_relevance_scores(query=query, k=k)

        # lo paso a dict porque el agente no se lleva bien con los objetos Document
        return [
            {"contenido": doc.page_content, "score": round(score, 3)}
            for doc, score in results
        ]
    except Exception as e:
        return f"Error al consultar la base vectorial: {e}"
