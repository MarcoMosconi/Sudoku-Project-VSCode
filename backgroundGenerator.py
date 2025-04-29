import base64
import io
import json
import logging
import argparse
import boto3
from PIL import Image
from botocore.exceptions import ClientError
from datetime import datetime
from gridTransformer import get_grid, process_sudoku_grid
import os
from pathlib import Path
from sudokuCorruptor import generate_invalid_sudoku

class ImageError(Exception):
    "Custom exception for errors returned by Amazon Nova Canvas"
    def __init__(self, message):
        self.message = message

def generate_image(model_id, body):
    """
    Generate an image using Amazon Nova Canvas model.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating image with Amazon Nova Canvas model %s", model_id)

    bedrock = boto3.client(service_name='bedrock-runtime')

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    
    response_body = json.loads(response.get("body").read())

    base64_image = response_body.get("images")[0]
    base64_bytes = base64_image.encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)

    finish_reason = response_body.get("error")
    if finish_reason is not None:
        raise ImageError(f"Image generation error. Error is {finish_reason}")

    logger.info("Successfully generated image")
    return image_bytes

def change_background(image_path, output_dir, grid_name):
    """
    Change image background using image conditioning.
    """
    try:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        
        # Model ID for Nova Canvas
        model_id = 'amazon.nova-canvas-v1:0'

        # Read and prepare input image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to supported dimensions if needed
            width = height = 512  # or 512 based on your needs
            #img = img.resize((width, height))
            
            # Convert to bytes and encode
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            input_image = base64.b64encode(img_byte_arr).decode('utf8')
        
        prompt = "brown or white desk with some objects, like a pen, a cup or a pc on the sides, respecting the perspective"
        
        # Prepare request body
        body = json.dumps({
     
            "taskType": "OUTPAINTING",
             "outPaintingParams": {
                "text": prompt,
                "negativeText": "bad quality, low resolution, blurry, distorted, out of focus",
                "image": input_image,
                "maskPrompt": "grid in the center",
                "outPaintingMode": "PRECISE",
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": height,
                "width": width,
                "cfgScale": 10,
                "seed": 12
            }
        })

        # Generate new image
        image_bytes = generate_image(model_id=model_id, body=body)
        
        # Save and show result
        output_image = Image.open(io.BytesIO(image_bytes))
        generation_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_dir, f"{Path(grid_name).stem}_{generation_id}.png")
        output_image.save(output_path)
        
        logging.info(f"Image saved as {output_path}")

    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
    except ClientError as err:
        logging.error(f"AWS service error: {err.response['Error']['Message']}")
    except ImageError as err:
        logging.error(err.message)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

def process_and_change_background(i, transform_folder, background_folder, prefix, generate_invalid=False):
    grid_name = f"{prefix}_{i}.png"
    grid = get_grid(transform_folder, grid_name)
    if grid is not None:
        if generate_invalid:
            grid = generate_invalid_sudoku(grid)
        result = process_sudoku_grid(grid, transform_folder, grid_name)
        print(f"Generated {prefix} {i}")
        try:
            change_background(result, background_folder, grid_name)
            print(f"Processed {prefix} {i}")
        except Exception as e:
            print(f"Failed to process {prefix} {i}: {e}")

def main():
    for i in range(300):
        process_and_change_background(i, 
                                      "transformed_images/valid_grids", 
                                      "changed_background/valid_grids", 
                                      "valid_grid")
        process_and_change_background(i, 
                                      "transformed_images/wrong_grids", 
                                      "changed_background/wrong_grids",
                                      "wrong_grid", 
                                      generate_invalid=True)

if __name__ == "__main__":
    main()