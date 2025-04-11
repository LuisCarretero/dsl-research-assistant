"""
Contains the definitions of the base classes used for the different tools
"""

import abc

class _BaseInferenceModel():
    """
    Base class for doing LLM based inference.
    Provides abstraction so multiple LLM packages can be used (e.g. HuggingFace or Ollama)
    """

    @abc.abstractmethod
    def predict(self, prompt:str) -> str:
        pass
    

class _BaseSuggestionGenerator():
    """
    Base class for generating suggestions for further text writing
    """
    @abc.abstractmethod
    def predict(self, existing_text:str, position_in_text:int) -> str:
        pass