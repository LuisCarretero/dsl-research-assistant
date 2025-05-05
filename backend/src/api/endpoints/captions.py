# from fastapi import APIRouter, UploadFile, File, Form
# from dsl_research_assistant.captions.image_captioning import ImageCaptioning
# import shutil
# import os
# from io import BytesIO
# from PIL import Image

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from src.dsl_research_assistant.captions.image_captioning import ImageCaptioning
import os
from io import BytesIO
from PIL import Image
import uuid
import logging

router = APIRouter()
caption_model = ImageCaptioning(model_name="Salesforce/blip-image-captioning-base", task="image-to-text")
# @router.post("/generate-caption/")
# async def generate_caption(
#     file: UploadFile = File(...),
#     prompt: str = Form(""),
#     context: str = Form("")
#     ):
#     """
#     Generate a caption for the uploaded image.
#     """
    
#     temp_path = f"temp_{file.filename}"
#     with open(temp_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     caption = caption_model.generate_caption(image_path=temp_path, prompt=prompt, context=context)
#     os.remove(temp_path)
#     return {"caption": caption}



@router.post("/generate-caption/")
async def generate_caption(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    context: str = Form("")
):
    """
    Generate a caption for the uploaded image.
    """
    # Validate file is an image
    
    try:
        # Reset file pointer
        await file.seek(0)
        
        # Read file into memory
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Save temporarily if your model requires a file path
        temp_path = f"temp_{uuid.uuid4()}.jpg"
        image.save(temp_path)
        
        # Generate caption
        caption = caption_model.generate_caption(
            image_path=temp_path, 
            prompt=prompt, 
            context=context
        )
        
        # Format response for the UI
        captions = [
            {
                "id": 1,
                "text": caption,
                "confidence": 95
            }
        ]
        
        return {"captions": captions}
    
    except Exception as e:
        logging.error(f"Error generating caption: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logging.error(f"Error removing temporary file: {str(e)}")