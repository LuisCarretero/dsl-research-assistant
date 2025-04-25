import abc

class Paper():
    def __init__(self, 
                 title:str,
                 authors:list[str]=None,
                 abstract:str=None,
                 sections:list[tuple[str, str]]=None,
                 references:list[str]=None):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.sections = sections
        self.references = references

    def from_markdown(path_to_paper:str, encoding:str="utf-8"):
        with open(path_to_paper, "r", encoding=encoding) as p:
            text = p.read()