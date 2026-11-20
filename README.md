# agent_adk_perez_anahi

Agente de IA hecho con **Google ADK** que ayuda con dudas de programación. Además de responder, puede guardar apuntes en una base de datos vectorial (Chroma) y después buscar ahí antes de salir a internet.

La idea es que no sea un chatbot más: el agente decide qué herramienta usar según lo que le pidas.

## ¿Qué hace?

- Si le pedís que **guarde** algo (un apunte, una explicación, un texto), lo indexa en la base vectorial.
- Si le **preguntás** algo, primero se fija qué colecciones hay, busca en la que corresponda y responde con eso.
- Si en la base no hay nada útil, **busca en internet** con DuckDuckGo.
- Si le preguntás **qué tiene guardado**, te lista las colecciones.

## Herramientas

| Tool | Qué hace | Archivo |
| --- | --- | --- |
| `web_search(query, max_results=5)` | Busca en internet con DuckDuckGo | `agent/agent.py` |
| `index_data(query, collection_name)` | Divide el texto en chunks y lo guarda en Chroma | `agent/tools.py` |
| `list_collection()` | Lista las colecciones de la base vectorial | `agent/tools.py` |
| `retrieval(query, collection_name, k=5)` | Busca los fragmentos más parecidos a la consulta | `agent/tools.py` |

### Validaciones

Todas las tools devuelven un mensaje claro en vez de romperse:

- **Argumentos inválidos:** consulta vacía, `max_results` o `k` menores o iguales a 0, nombre de colección inválido (Chroma pide de 3 a 63 caracteres, sin espacios).
- **Resultados vacíos:** si no hay resultados en internet, si no hay colecciones o si la búsqueda en la base no trae nada, se avisa para que el modelo no invente.
- **Errores:** cualquier excepción (sin conexión, Ollama apagado, etc.) se captura y se devuelve como texto.
- Además, `retrieval` revisa que la colección exista antes de consultarla, así no se crean colecciones vacías por error.

## Tecnologías

- Google ADK
- LiteLLM + Ollama (modelo `gemma4:31b-cloud`)
- `nomic-embed-text` para los embeddings
- ChromaDB + LangChain
- ddgs (DuckDuckGo Search)

## Cómo usarlo

### 1. Requisitos

- Python 3.10 o superior
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) instalado y corriendo

Bajar el modelo de embeddings (y el de chat si no lo tenés):

```shell
ollama pull nomic-embed-text
ollama pull gemma4:31b-cloud
ollama ls
```

### 2. Clonar e instalar

```shell
git clone https://github.com/TU_USUARIO/agent_adk_nombre_apellido.git
cd agent_adk_nombre_apellido

uv venv env
env\Scripts\activate        # windows
source env/bin/activate     # linux / mac

uv pip install -r requirements.txt
```

### 3. Ejecutar

Parado en la carpeta del proyecto (no adentro de `agent/`):

```shell
adk web
```

Abrir `http://localhost:8000` y elegir `agent` en el selector de arriba a la izquierda.

## Estructura

```
agent_adk_nombre_apellido/
├── agent/
│   ├── __init__.py
│   ├── agent.py        # agente + web_search
│   ├── tools.py        # index_data, list_collection, retrieval
│   └── .env
├── requirements.txt
├── .gitignore
└── README.md
```
