from pathlib import Path
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

from typing import Iterable, Union

import re
import numpy as np
from writing_tools._base import _BaseInferenceModel
from writing_tools import OllamaInferenceModel
from pyalex import Works
from tqdm import tqdm

    
'''
class Figure():
    def __init__(self,
                 number:int,
                 caption:str=None,
                 caption_position:str="above figure",
                 image:np.ndarray=None,
                 position_in_text:int=0):
        self.number = number
        self.caption = caption
        self.caption_position = caption_position
        self.image = image
        self.position_in_text = position_in_text


class Table():
    def __init__(self,
                 number:int,
                 caption:str=None,
                 caption_position:str="above",
                 content:list=None,
                 position_in_text:int=-1):
        self.number = number
        self.caption = caption
        self.caption_position = caption_position
        self.content = content
        self.position_in_text = position_in_text


class Paper2():
    def __init__(self, 
                 title:str,
                 authors:list[str]=None,
                 abstract:str=None,
                 content:str=None,
                 figures:list[Figure]=None,
                 tables:list[Table]=None,
                 references:list[str]=None,
                 doi:str=None,
                 publication_year:int=None):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.content = content
        self.figures = figures
        self.tables = tables
        self.references = references
        self.doi = doi
        self.publication_year = publication_year

    @staticmethod
    def from_docling_markdown_file(path_to_paper:str, encoding:str="utf-8") -> "Paper":
        with open(path_to_paper, "r", encoding=encoding) as p:
            text = p.read()
        return Paper._from_docling_markdown_text(text)

    @staticmethod
    def from_openalex_hash(openalex_hash:str):
        try:
            work = Works()[openalex_hash]
            title = work["title"]
        except:
            return None 

    @staticmethod
    def _from_docling_markdown_text(text:str, inference_model:_BaseInferenceModel=OllamaInferenceModel()) -> "Paper":
        # Get the first 2000 characters to extract the title and authors if needed
        starting_text = text[:2000]
        query = f"""
        Find the title of the research paper given the following text:

        {starting_text}

        Only provide the title in your output, do not include any additional information.
        Before the output, write the keyword "Title:"
        """
        response = inference_model.predict(query).replace("\n", "")
        pos = list(re.finditer("Title:", response))[-1].end()
        paper_title = response[pos:]

        # Try finding the paper with Open Alex
        doi = None
        publication_year = None
        authors = None
        references = None
        abstract = None
        try:
            openalex_paper = Works().search(paper_title).get()[0]
            doi = openalex_paper["doi"] if "doi" in openalex_paper else None
            paper_title = openalex_paper["title"]
            publication_year = openalex_paper["publication_year"] if "publication_year" in openalex_paper else None
            authors = [i["author"]["display_name"] for i in openalex_paper["authorships"]]
            print(authors)
            abstract = openalex_paper["abstract"] if "abstract" in openalex_paper else None
            references = [Works()[work]["doi"] for work in openalex_paper["referenced_works"]] if openalex_paper["referenced_works_count"] > 0 else None
        except:
            pass
        # If the information is not available, infer it with the inference model
        #if authors is None:
        #    authors = extract_authors_from_paper_text(text, inference_model=inference_model)
        #if references is None:
        #    references = extract_references_from_paper_text(text, inference_model=inference_model)
        if abstract is None:
            print("Finding abstract automatically...")
            abstract = re.findall("## Abstract([^#]*)##", text)[0]
            abstract = abstract.replace("\n\n", "")


        # Extract figure captions
        figure_captions = re.findall("\n\nFigure [0-9]+.*\n\n", text)
        # Extract table captions
        table_captions = re.findall("\n\nTable [0-9]+.*\n\n", text)
        # Extract tables
        tables_text = re.findall("\n\n[|][\s\S]*[|]\n\n", text)

        # Remove everything that was extracted and store it
        figures = []
        for i in range(len(figure_captions)):
            figure_caption = figure_captions[i]
            figures.append(Figure(
                i,
                caption=figure_caption,
                caption_position="above"
            ))
        tables = []
        for i in range(max(len(table_captions), len(tables_text))):
            table_caption = table_captions[i] if i < len(table_captions) else ""
            table_text = tables_text[i] if i < len(tables_text) else ""
            table = Table(
                i,
                caption=table_caption,
                content=table_text
            )
            tables.append(table)

    
        paper = Paper2(
            paper_title,
            authors = authors,
            abstract = abstract,
            content = text,
            references = references,
            figures=figures,
            tables=tables,
            doi = doi
        )

        return paper
'''

class Paper():
    def __init__(self, 
                document:DoclingDocument,
                title:Union[str, None] = None,
                doi:Union[str, None] = None,
                publication_year:Union[int, None] = None,
                authors:Union[Iterable[str], None] = None,
                abstract:Union[str, None] = None,
                inference_model:_BaseInferenceModel=OllamaInferenceModel()):
        self.document = document
        self.inference_model = inference_model
        self.doi = doi
        if isinstance(self.doi, str):
            try:
                self.pyalex_work = Works()[self.doi]
            except:
                self.pyalex_work = None
        self.title = self._find_title(title)
        if self.pyalex_work is None:
            try:
                self.pyalex_work = Works().search(self.title).get()[0]
                self.doi = self.pyalex_work["doi"]
            except:
                self.pyalex_work = None
        self.abstract = self._find_abstract(abstract)
        self.publication_year = self._find_publication_year(publication_year)
        self.authors = self._find_authors(authors)

    def _find_title(self, title:Union[str, None]) -> Union[str, None]:
        if isinstance(title, str): return title
        if self.pyalex_work is not None: return self.pyalex_work["title"]
        # If title was not given, find it with an LLM
        text = self.document.export_to_markdown()
        starting_text = text[:2000]
        query = f"""
        Find the title of the research paper given the following text:

        {starting_text}

        Only provide the title in your output, do not include any additional information.
        Before the output, write the keyword "Title:"
        """
        response = self.inference_model.predict(query).replace("\n", "")
        pos = list(re.finditer("Title:", response))[-1].end()
        paper_title = response[pos:]
        return paper_title

    def _find_publication_year(self, publication_year:Union[int, None]) -> Union[int, None]:
        if isinstance(publication_year, int): return publication_year
        if self.pyalex_work is not None: return self.pyalex_work["publication_year"]
        return None

    def _find_authors(self, authors:Union[Iterable[str], None]) -> Union[list[str], None]:
        if isinstance(authors, Iterable): return [author for author in authors]
        if self.pyalex_work is not None: return [author["author"]["display_name"] for author in self.pyalex_work["authorships"]]
        return None

    def _find_abstract(self, abstract:Union[str, None]) -> Union[str, None]:
        if isinstance(abstract, str): return abstract
        if self.pyalex_work is not None:
            if "abstract" in self.pyalex_work and self.pyalex_work["abstract"] is not None:
                return self.pyalex_work["abstract"]
        return None

    def get_section_text(self, section:str) -> str:
        for item in self.document.iterate_items():
            if item[0].dict()['label'] == 'section_header':
                print(item[0].dict())

    @staticmethod
    def from_document_path(path:Union[Path,str,DocumentStream], **kwargs) -> "Paper":
        converter = DocumentConverter()
        result = converter.convert(path)
        return Paper(result.document, **kwargs)

    @staticmethod
    def from_docling_document(document:DoclingDocument, **kwargs) -> "Paper":
        return Paper(document, **kwargs)

    @staticmethod
    def from_docling_json(path:Union[Path, str], **kwargs) -> "Paper":
        document = DoclingDocument.load_from_json(path)
        return Paper(document, **kwargs)

    @staticmethod
    def from_docling_doctags(path:Union[Path, str], **kwargs) -> "Paper":
        document = DoclingDocument.load_from_doctags(path)
        return Paper(document, **kwargs)

    @staticmethod
    def from_docling_yaml(path:Union[Path, str], **kwargs) -> "Paper":
        document = DoclingDocument.load_from_yaml(path)
        return Paper(document, **kwargs)
'''
def extract_references_from_paper_text(paper_text:str, inference_model:_BaseInferenceModel=OllamaInferenceModel()):
    print("Finding references automatically...")
    # Fetch the references automatically
    last_section = paper_text.split("##")[-1]
    # All references as a list
    full_references = last_section.split("\n")
    references = []
    for full_reference in tqdm(full_references):
        refernece_prompt = f"""
        Extract the title of the given reference taken from a research paper:

        {full_reference}

        In your output, include only the title and no other information.
        Before your output, write the keyword "Title:". In case the provided information does not contain a reference, simply output "Wrong input format"
        """
        response = inference_model.predict(refernece_prompt)
        if "Wrong input format" not in response:
            pos = list(re.finditer("Title:", response))[-1].end()
            ref_title = response[pos:].strip()
            references.append(ref_title)
            print(ref_title)
    return references


def extract_authors_from_paper_text(paper_text:str, inference_model:_BaseInferenceModel=OllamaInferenceModel()):
    starting_text = paper_text[:2000]

    print("Finding authors automatically...")
    author_prompt = f"""
    Find the authors of the research paper given the following text:

    {starting_text}

    Only provide the authors names, without the meta-data such as their affeliations.
    Output the names as a comma-sepparated list. Do not provide any other output. Before the output,
    write the keyword "Authors:"
    """
    response = inference_model.predict(author_prompt).replace("\n", "")
    pos = list(re.finditer("Authors:", response))[-1].end()
    authors = [author.strip() for author in response[pos:].split(",")]
'''

if __name__ == '__main__':
    from dotenv import load_dotenv
    import os

    load_dotenv()

    DATA_DIR = os.environ.get("DATA_DIR")

    path = os.path.join(DATA_DIR, "Conversions\\opencvf-data\\md\\Abouee_Weakly_Supervised_End2End_Deep_Visual_Odometry_CVPRW_2024_paper.md")

    paper = Paper.from_docling_markdown_file(path, encoding="utf-8")

    print(paper.title)
    print(paper.authors)
    print(paper.abstract)
    print(paper.references)
    print()
    for table in paper.tables:
        print(table.caption)
        print(table.content)



    #works = Works().search("""Dasdčlkhačlkahsdlkfhlsakdjfhlkjhaf sldakjhflk """).get()
    #print(works)