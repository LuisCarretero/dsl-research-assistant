"""
Contains the definitions of the base classes used for the different tools
"""

import abc
from typing import Union, Dict, List

class _BaseInferenceModel():
    """
    Base class for doing LLM based inference.
    Provides abstraction so multiple LLM packages can be used (e.g. HuggingFace or Ollama)
    """
    default_call_kwargs = {}
    
    def postprocess_output(self, output:str) -> str:
        return output

    def predict(self, user_prompt:str, system_prompt:Union[str, None]=None, **call_kwargs) -> str:
        for key in self.default_call_kwargs.keys():
            if key not in call_kwargs.keys():
                call_kwargs[key] = self.default_call_kwargs[key]
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.postprocess_output(self._predict(messages, **call_kwargs))
    
    def set_default_call_kwargs(self, **default_call_kwargs):
        self.default_call_kwargs = default_call_kwargs

    def postprocess_output(self, output:str):
        return output

    @abc.abstractmethod
    def _predict(self, messages:List[Dict[str, str]], **call_kwargs) -> str:
        pass
    

class _BaseSuggestionGenerator():
    """
    Base class for generating suggestions for further text writing
    """
    @abc.abstractmethod
    def predict(self, existing_text:str, position_in_text:int, citation_context:list[str]) -> str:
        pass


class _BaseLiteratureReviewGenerator():
    """
    Base class for generating literature reviews
    """
    @abc.abstractmethod
    def predict_entire(self, query:str, references:list[str], reference_ids:list[int], **kwargs) -> str:
        pass

    @abc.abstractmethod
    def predict_next(self, query:str, references:list[str], reference_ids:list[int], related_work_draft:str, **kwargs) -> str:
        pass