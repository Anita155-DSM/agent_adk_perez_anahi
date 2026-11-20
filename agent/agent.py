import os

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"


root_agent = Agent(
    model=LiteLlm("ollama_chat/qwen3:4b"),
    name='root_agent',
    description='Un asistente util que responde preguntas',
    instruction='Eres un asisistente util que responde preguntas sobre programacion.',
)
