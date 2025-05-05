import logging
from pathlib import Path
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
import re
from typing import List
import os
import tqdm
from transformers import pipeline
from PIL import Image
import torch
from typing import Optional

class ImageCaptioning:
    """
    A class to handle image captioning tasks.
    """

    def __init__(self, model_name: str, task: str):
        """
        Initializes the ImageCaptioning class with a specified model name.

        Args:
            model_name (str): The name of the image captioning model to use.
        """
        # Load model and processor
        self.resolution_scale = 2.0
        self.model_name = model_name
        self.task = task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(model_name, task) 
        
    def load_model(self, model_name: str, task: str = ""):
        """
        Loads the specified image captioning model and tokenizer.

        Args:
            model_name (str): The name of the model to load.
        
        Returns:
            tuple: A tuple containing the loaded model and tokenizer.
        """
        self.pipe = pipeline(task=task, model=model_name, device=self.device, use_fast=True)

    def export_images_from_paper(self, paper_paths: List[str]):
        """
        Extracts images from a given research paper.

        Args:
            paper_path (str): The path to the research paper.

        Returns:
            list: A list of paths to the extracted images.
        """
        logging.basicConfig(level=logging.INFO)
        accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.CUDA,
        )
        pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)
        pipeline_options.images_scale = self.resolution_scale
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        for paper_path in tqdm.tqdm(paper_paths, desc="Processing papers"):
            input_doc_path = Path(paper_path)
            conv_res = doc_converter.convert(input_doc_path)
            doc_filename = conv_res.input.file.stem
            output_dir = Path("image_captions/"+doc_filename)
            output_dir.mkdir(parents=True, exist_ok=True)

            table_counter = 0
            picture_counter = 0
            previous_text = ""
            for element, _level in conv_res.document.iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    element_image_filename = (
                        output_dir / f"{doc_filename}-table-{table_counter}.png"
                    )
                    with element_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")

                if isinstance(element, PictureItem):
                    picture_counter += 1
                    element_image_filename = (
                        output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                    )
                    with element_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")
                        caption = element.caption_text(conv_res.document)
                        with open("image_captions/"+doc_filename+"/texts.txt", "a", encoding='utf-8') as texts_file:
                            print(previous_text, file=texts_file)
                            print("________________________________________", file=texts_file)
                            
                        cleaned = re.sub(r'^Figure\s*\d+\.\s*', '', caption)
                        with open("image_captions/"+doc_filename+"/caption.txt", "a", encoding='utf-8') as caption_file:
                            print(cleaned, file=caption_file)
                            print("________________________________________", file=caption_file)
                if isinstance(element, TextItem):
                    previous_text += " " + element.text
                    previous_text = previous_text.replace("\n", " ")
                    previous_text = " ".join(previous_text.split()[-100:])
        pass

    def generate_caption(self, image_path: str, prompt: str = "", context: Optional[str] = "") -> str:
        """
        Generates a caption for the given image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The generated caption for the image.
        """
        image = Image.open(image_path).convert("RGB")
        if self.task == "image-to-text":
            caption = self.pipe(
                image,
            )[0]['generated_text'].strip()
        elif self.task == "image-text-to-text":
            prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")
            caption = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})[0]['generated_text'].strip()
        return caption

if __name__ == "__main__":
    image_captioning = ImageCaptioning(model_name="ds4sd/SmolDocling-256M-preview")
    
    # pdf_files = []
    # for root, dirs, files in os.walk("data"):
    #     for file in files:
    #         if file.endswith(".pdf"):
    #             pdf_files.append(os.path.join(root, file))
    # images = image_captioning.export_images_from_paper(pdf_files[:30])
    
    