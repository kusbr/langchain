from typing import Dict, List, Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

class AgentCallbackHandler(BaseCallbackHandler):
	def __init__(self):
		super().__init__()

	# override the callback method on_llm_start
	def on_llm_start(self, 
			serialized: dict[str, Any],
        	prompts: list[str],
			**kwargs: Any
	) -> Any:
		print(f"***Prompt to LLM:***\n{prompts}")	
		print(f"**************")

	# override the callback method on_llm_end
	def on_llm_end(self, 
			response: LLMResult,
			**kwargs: Any
	) -> Any:
		print(f"***LLM Response:***\n{response}")	