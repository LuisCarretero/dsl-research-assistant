from docling.document_converter import DocumentConverter


# Class to encapsulate parsing of documents 
# - retrieval of documents as dictionary where keys are specific sections of the paper (example: Introduction, Methodology)
# - values are texts of given sections
class DocumentParser():
    def __init__(self):
        self.document_converter = DocumentConverter()
    
    def load_docling(self, file_name: str):
        result = self.document_converter.convert(file_name)
        return result.document
    
    def doc_to_dict(self, docling_document):
        return docling_document.export_to_dict()
    def doc_to_markdown(self, docling_document):
        return docling_document.export_to_markdown()

parser = DocumentParser()

doc = parser.load_docling("data\CVPR_2024\Workshops\Abouee_Weakly_Supervised_End2End_Deep_Visual_Odometry_CVPRW_2024_paper.pdf")
print(doc)