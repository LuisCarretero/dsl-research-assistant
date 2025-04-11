import logging
import time
from pathlib import Path
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

class ImageCaptioning:
    """
    A class to handle image captioning tasks.
    """

    def __init__(self, model_name: str):
        """
        Initializes the ImageCaptioning class with a specified model name.

        Args:
            model_name (str): The name of the image captioning model to use.
        """
        self.resolution_scale = 2.0
        self.model_name = model_name
        # Load the model here (pseudo code)
        # self.model = load_model(model_name)
    
    def export_images_from_paper(self, paper_path: str) -> list:
        """
        Extracts images from a given research paper.

        Args:
            paper_path (str): The path to the research paper.

        Returns:
            list: A list of paths to the extracted images.
        """
        logging.basicConfig(level=logging.INFO)

        input_doc_path = Path(paper_path)
        output_dir = Path("scratch")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self.resolution_scale
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        start_time = time.time()

        conv_res = doc_converter.convert(input_doc_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        doc_filename = conv_res.input.file.stem

        # Save page images
        for page_no, page in conv_res.document.pages.items():
            page_no = page.page_no
            page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")

        # Save images of figures and tables
        table_counter = 0
        picture_counter = 0
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

        # Save markdown with embedded pictures
        md_filename = output_dir / f"{doc_filename}-with-images.md"
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

        # Save markdown with externally referenced pictures
        md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

        # Save HTML with externally referenced pictures
        html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
        conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

        end_time = time.time() - start_time
        print(end_time)
        return ["image1.png", "image2.png"]

    def generate_caption(self, image_path: str) -> str:
        """
        Generates a caption for the given image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The generated caption for the image.
        """
        # Load and preprocess the image (pseudo code)
        # image = load_image(image_path)
        # caption = self.model.generate_caption(image)
        # return caption
        return "Generated caption for the image."  # Placeholder implementation
    

if __name__ == "__main__":
    image_captioning = ImageCaptioning(model_name="example_model")
    paper_path = "data\CVPR_2024\Workshops\Abouee_Weakly_Supervised_End2End_Deep_Visual_Odometry_CVPRW_2024_paper.pdf"
    images = image_captioning.export_images_from_paper(paper_path)
    for image in images:
        caption = image_captioning.generate_caption(image)
        print(f"Caption for {image}: {caption}")
    