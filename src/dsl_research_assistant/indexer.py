from llama_index import GPTSimpleVectorIndex, Document
from dsl_research_assistant.data_preprocessing import DocumentParser
import os


# Class encapsulates creation and usage of Vector Database. Possble VD to use are Milvus, Qdrant ...
class VectorDatabase():
    def __init__(self):
        self.parser = DocumentParser()
        
    def create_index(self, data_dir):
        documents = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(data_dir, filename)
                doc = self.parser.load_docling(file_path)
                documents.append(Document(doc))
        index = GPTSimpleVectorIndex.from_documents(documents)
        index.save_to_disk("papers_index.json")