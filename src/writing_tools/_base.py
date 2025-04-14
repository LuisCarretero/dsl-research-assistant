"""
Contains the definitions of the base classes used for the different tools
"""

import abc
from typing import Union

class _BaseInferenceModel():
    """
    Base class for doing LLM based inference.
    Provides abstraction so multiple LLM packages can be used (e.g. HuggingFace or Ollama)
    """

    @abc.abstractmethod
    def predict(self, prompt:str, **kwargs) -> str:
        pass
    

class _BaseSuggestionGenerator():
    """
    Base class for generating suggestions for further text writing
    """
    @abc.abstractmethod
    def predict(self, existing_text:str, position_in_text:int, citation_context:list[str]) -> str:
        pass