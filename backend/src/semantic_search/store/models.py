from transformers import AutoTokenizer, AutoModel
import torch
from typing import Union, Literal, Dict, Any, List, Optional, Callable
import numpy as np
from tqdm import tqdm
import sys
from transformers.tokenization_utils_base import BatchEncoding
import os


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class LocalEmbeddingModel:
    def __init__(
        self, 
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        batch_size: int = 8,
        device: str | None = None,
        pooling_type: Literal['mean', 'last', 'cls'] = 'mean',  # cls is first token
        normalize_embeddings: bool = True,
        preferred_index_metric: Literal['l2', 'ip'] = 'l2',
        query_formatter: Optional[Callable[[str], str]] = None,
        ingestion_formatter: Optional[Callable[[str], str]] = None
    ):
        # Tokenizer and model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Settings/config
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_dim = self.model.config.hidden_size
        assert pooling_type in ['mean', 'last', 'cls'], f"Invalid pooling type: {pooling_type}"
        self.pooling_type = pooling_type
        self.normalize_embeddings = normalize_embeddings
        self.preferred_index_metric = preferred_index_metric
        self.query_formatter = query_formatter
        self.ingestion_formatter = ingestion_formatter

        # Device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = torch.device(device)
        print(f'Using device: {self.device}')
        self.model.to(self.device)

    def chunk_and_encode(self, texts: Union[list[str], str], progress_bar: bool = False, is_query: bool = False) -> tuple[list[str], list[dict]]:
        """
        Chunk texts at token level using the model's tokenizer and encode the chunks.

        Didnt manage to call tokenizer.prepare_for_model batched so doing weird back and forth
        conversion for now.

        Returns:
            tuple of:
                - all_chunks_text: list of lists of strings, each string is a chunk of the original text, 
                grouped into lists that constitute a single input text
                Example: [ ['text1_chunk1', 'text1_chunk2'], ['text2_chunk1', 'text2_chunk2', 'text2_chunk3'] ]
                - all_chunks_encoded: list of dicts, each dict is a batch of encoded chunks

        TODO: Allow for np arrays or other iterable inputs.
        TODO: Make this more efficient!
        """
        if isinstance(texts, str):
            texts = [texts]
            unwrap = True
        else:
            assert isinstance(texts, list), "texts must be a list"
            unwrap = False
        all_chunks_text, all_chunks_encoded = [], []

        if is_query and self.query_formatter is not None:
            texts = [self.query_formatter(text) for text in texts]
        elif not is_query and self.ingestion_formatter is not None:
            texts = [self.ingestion_formatter(text) for text in texts]

        effective_chunk_size = self.chunk_size - 2  # Subtract 2 for the special tokens
        stride = effective_chunk_size - self.chunk_overlap

        # Temporarily increase model_max_length to supress warning message.
        tmp, self.tokenizer.model_max_length = self.tokenizer.model_max_length, sys.maxsize
        texts_tokenized = self.tokenizer(
            texts, 
            add_special_tokens=False, 
            return_token_type_ids=False, 
            return_attention_mask=False,
            padding=False,
        )
        self.tokenizer.model_max_length = tmp

        for text_tokenized in tqdm(texts_tokenized['input_ids'], desc='Chunking and encoding', disable=not progress_bar):
            # Create chunks of tokens
            text_chunks = []
            for i in range(0, len(text_tokenized), stride):
                chunk = text_tokenized[i:i+effective_chunk_size]
                if len(chunk) > 0:  # Only keep non-empty chunks
                    chunk_text = self.tokenizer.decode(chunk)
                    text_chunks.append(chunk_text)

            # Batch encode all chunks for this text
            if text_chunks:
                encoded_chunks = self.tokenizer(
                    text_chunks,
                    padding='max_length',
                    max_length=self.chunk_size,
                    truncation=True,
                    return_tensors='pt'
                )
            else:
                encoded_chunks = None
            
            # Also add to flat lists for backward compatibility
            all_chunks_text.append(text_chunks)
            all_chunks_encoded.append(encoded_chunks)
        
        if unwrap:
            return all_chunks_text[0], all_chunks_encoded[0]
        return all_chunks_text, all_chunks_encoded
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        Same operation as sentence_transformers.model.Pooling(ndim, pooling_mode='mean')
        """
        if attention_mask is None:
            return torch.mean(token_embeddings, dim=1)
        else:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def _flatten_dict(self, dicts: List[dict]) -> dict:
        tmp = {}
        for single_text_encoded in dicts:
            for k, v in single_text_encoded.items():
                if k not in tmp:
                    tmp[k] = []
                tmp[k].append(v)
        return {k: torch.cat(v) for k, v in tmp.items()}

    def get_embeddings(self, encoded_inputs: BatchEncoding | List[BatchEncoding], progress_bar: bool = False) -> np.ndarray:
        """Generate embeddings from pre-tokenized inputs.
        
        Args:
            encoded_inputs: BatchEncoding object from tokenizer.prepare_for_model
            progress_bar: Whether to show a progress bar

        Returns:
            np.ndarray of shape (n_texts, embedding_dim)
        """

        if isinstance(encoded_inputs, list):
            if isinstance(encoded_inputs[0], dict) or isinstance(encoded_inputs[0], BatchEncoding):
                # List of batchencodings
                encoded_inputs = self._flatten_dict(encoded_inputs)
            else:
                raise ValueError(f"Invalid encoded_inputs type: List of {type(encoded_inputs[0])}")
        elif not (isinstance(encoded_inputs, dict) or isinstance(encoded_inputs, BatchEncoding)):
            raise ValueError(f"Invalid encoded_inputs type: {type(encoded_inputs)}")

        embeddings = []
        for i in tqdm(range(0, len(encoded_inputs['input_ids']), self.batch_size), desc="Generating embeddings", disable=not progress_bar):
            batch_encoded = {k: v[i:i+self.batch_size] for k, v in encoded_inputs.items()}

            batch_dict = {k: v.to(self.device) for k, v in batch_encoded.items()}
            
            with torch.no_grad():
                model_output = self.model(**batch_dict)

            if self.pooling_type == 'mean':
                pooled = self._mean_pooling(model_output['last_hidden_state'], batch_dict['attention_mask'])
            elif self.pooling_type == 'last':
                raise NotImplementedError("Last pooling not implemented")  
                # FIXME: Check if this needs attention_mask to not index padding values.
                pooled = model_output['last_hidden_state'][:, -1, :]
            elif self.pooling_type == 'cls':
                pooled = model_output['last_hidden_state'][:, 0, :]
            else:
                raise ValueError(f"Invalid pooling type: {self.pooling_type}")
            
            if self.normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            
            embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the model needed for initializing."""
        return {
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'batch_size': self.batch_size,
            'pooling_type': self.pooling_type,
            'normalize_embeddings': self.normalize_embeddings,
            'preferred_index_metric': self.preferred_index_metric
        }


DEFAULT_MODEL_PARAMS = {
    'sentence-transformers/all-MiniLM-L6-v2': dict(
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        chunk_size=256,
        chunk_overlap=32,
        batch_size=8,
        pooling_type='mean',
        normalize_embeddings=True,
        preferred_index_metric='ip'
    ),
    'allenai/specter2_base': dict(
        # https://huggingface.co/allenai/specter2
        chunk_size=512,
        chunk_overlap=64,
        batch_size=16,
        pooling_type='cls',
        normalize_embeddings=True,
        preferred_index_metric='ip'
    ),
    'prdev/mini-gte': dict(
        # https://huggingface.co/prdev/mini-gte
        chunk_size=512,
        chunk_overlap=64,
        batch_size=8,
        pooling_type='mean',  # Used HF config to infer this.
        normalize_embeddings=True,
        preferred_index_metric='ip'
    ),
    'ibm-granite/granite-embedding-278m-multilingual': dict(
        # https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual
        chunk_size=512,
        chunk_overlap=64,
        batch_size=4,
        pooling_type='cls',
        normalize_embeddings=True,
        preferred_index_metric='ip'
    ),
    # 'Snowflake/snowflake-arctic-embed-m-v2.0': dict( <- Requires xformers, not possible on MacM1
    #     # https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0
    #     chunk_size=8192,
    #     chunk_overlap=512,
    #     batch_size=1,
    #     pooling_type='cls',
    #     normalize_embeddings=True,
    #     preferred_index_metric='ip',
    #     query_formatter=lambda x: f"query:{x}"
    # ),
    'Alibaba-NLP/gte-multilingual-base': dict(  # <- Too large for MacM1
        # https://huggingface.co/Alibaba-NLP/gte-multilingual-base
        chunk_size=8192,
        chunk_overlap=512,
        batch_size=1,
        pooling_type='cls',
        normalize_embeddings=True,
        preferred_index_metric='ip',
    )
}


def create_embedding_model(model_name: str, device: Optional[str] = None) -> LocalEmbeddingModel:
    """
    Create an embedding model from a model name using default parameters.
    """
    if model_name not in DEFAULT_MODEL_PARAMS:
        raise ValueError(f"Invalid model name: {model_name}. Available models: {DEFAULT_MODEL_PARAMS.keys()}")
    
    params = DEFAULT_MODEL_PARAMS[model_name]
    if device is not None:
        params['device'] = device

    return LocalEmbeddingModel(model_name, **params)
