from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict
from pydantic import BaseModel
import json
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
import os
import datetime
from fastapi.middleware.cors import CORSMiddleware
from biobert_ner import BioBertNER

# Initialize BioBERT NER model
bio_ner = BioBertNER()

# Initialize FastAPI app
app = FastAPI()

# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')


# Define Pydantic model for JSON input validation
class InputData(BaseModel):
    data: Dict


def find_box_center(coordinates):
    """
    Find the center coordinates of a box given its corner points.
    """
    x_coords = [point[0] for point in coordinates]
    y_coords = [point[1] for point in coordinates]
    center_x = (min(x_coords) + max(x_coords)) / 2
    center_y = (min(y_coords) + max(y_coords)) / 2
    return [center_x, center_y]


def bio_tagger(ocr_items, fields=None):
    """Tag OCR items using the local BioBERT NER model.

    Args:
        ocr_items: list of tuples (id, text, coord)
        fields: unused placeholder to keep compatibility with previous API

    Returns:
        list of tuples (id, text, entity_label)
    """
    return bio_ner.tag(ocr_items)

def process_image_with_ocr(img_path, output_image_path, fields, modee):
    """
    Perform OCR on the image, annotate results, and return extracted data.
    """
    result = ocr.ocr(img_path, cls=True)
    text_content = []

    print("Result: ", result)

    for idx, item in enumerate(result[0], start=1):
        coordinates, (text, confidence) = item
        text_content.append((idx, text, find_box_center(coordinates)))

    # Tag text using BioBERT
    ner_response = None
    if modee == 1:
        while ner_response is None:
            ner_response = bio_tagger(text_content, fields)

    # Open the image and draw the results
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size=12)

    data = []
    database = []
    ser = 1
    for i, res in enumerate(result[0], start=1):
        coordinates, (text, confidence) = res
        if modee == 1:
            label = ner_response[i - 1][2]
        else:
            label = '0'
        database.append({

            "id": i,
            "entityValue": text,
            "entityName": label,
            "pixelCoord": coordinates
        }, )
        if label != '0':
            data.append({
                "serial no.": ser,
                "id": i,
                "entityValue": text,
                "entityName": label,
                "pixelCoord": coordinates
            }, )
            ser += 1
        box = [tuple(point) for point in res[0]]
        box_coords = [(min(pt[0] for pt in box), min(pt[1] for pt in box)),
                      (max(pt[0] for pt in box), max(pt[1] for pt in box))]

        # Draw bounding box and text
        draw.rectangle(box_coords, outline="red", width=2)
        if label != '0':
            if label.startswith("full_disease_name_"):
                label = label.replace("full_disease_name_", "Disease:")
            elif label.startswith("drug_"):
                label = label.replace("drug_", "Drug:")
            draw.text((box_coords[0][0], box_coords[0][1] - 15), f"{label}", fill="blue", font=font)
    json.dump(database, open(f"{output_image_path}.json", 'w', encoding='utf-8'), indent=4)
    # Save the annotated image
    image.save(output_image_path)

    return data

def clean_entity_values(response):
    for item in response["data"]:
        entity_value = item["entityValue"]
        field = item["entityName"]
        
        # Check if any field is present in the entity value
        if field in entity_value:
            # Strip the field from the entity value
            item["entityValue"] = entity_value.replace(field, "").strip()
            break
    
    return response

@app.post("/upload/")
async def upload_file(
        file: UploadFile = File(...),
        json_data: str = Form(...),
):
    # Step 1: Read the contents of the uploaded file
    file_content = await file.read()

    # Save the file temporarily for processing
    temp_file_path = f"./tmp/{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(file_content)
    # Step 2: Parse the JSON data
    try:
        parsed_json_data = json.loads(json_data)  # example json-{"fields":["name","date"],"mode":"1"}
        json_dict = InputData(data=parsed_json_data).dict()
        fields = json_dict["data"]["fields"]
        print(json_dict)
        try:
            modee = int(json_dict["data"]["mode"])  # mode = 1 for automated tagging
            # modee=int(input("1 for automated tagging and 0 for manual tagging:"))
        except:
            modee = 1
            print("defaulting to automated tagging")

    except json.JSONDecodeError:
        return JSONResponse(content={"message": "Invalid JSON data"}, status_code=400)

    # Step 3: Process the image with OCR and extract fields
    output_img_path = f"./tmp/annotted_{file.filename}"
    try:
        data = process_image_with_ocr(temp_file_path, output_img_path, fields, modee)  # mode=1 for automated tagging
    except Exception as e:
        return JSONResponse(content={"message": f"OCR processing failed: {str(e)}"}, status_code=500)

    # Step 4: Prepare the response
    response = {
        "fields": fields,
        "data": data,
    }
    print("ner_response---", response)

    #cleaned_response = clean_entity_values(response, response["fields"])

    return JSONResponse(content=response)


from typing import List, Dict, Any

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AreaFilterResponse(BaseModel):
    status: str
    filtered_data: List[Dict[Any, Any]]
    total_matches: int


def is_point_in_area(point, area_coords):
    """
    Check if a point lies within a given area using ray casting algorithm
    """
    x, y = point
    n = len(area_coords)
    inside = False

    p1x, p1y = area_coords[0]
    for i in range(n + 1):
        p2x, p2y = area_coords[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def filter_data_in_area(data, area_coords):
    """
    Filter JSON records that have any point within the specified area
    """
    filtered_data = []

    for item in data:
        pixel_coords = item['pixelCoord']
        for point in pixel_coords:
            if is_point_in_area((point[0], point[1]), area_coords):
                filtered_data.append(item)
                break

    return filtered_data


@app.post("/check/")
async def upload_file(
        json_data: str = Form(...),
):
    """
    Upload endpoint for file and area coordinates

    Args:
        file: JSON file containing annotation data
        json_data: JSON string containing area coordinates

    Returns:
        JSON response with filtered data and metadata
    """
    try:
        # Parse the area coordinates from json_data
        area_coords = json.loads(json_data)
        print(area_coords)
        file_name = area_coords['file_name']
        print(file_name)
        areaa_coords = area_coords['json_data']

        # Validate area coordinates format
        if not isinstance(areaa_coords, list) or len(areaa_coords) < 3:
            raise HTTPException(
                status_code=400,
                detail="Area coordinates must be a list of at least 3 points"
            )

        with open(f"./tmp/annotted_{file_name}.json", "r") as file:
            file_data = json.loads(file.read())

        # Filter the data
        filtered_results = filter_data_in_area(file_data, areaa_coords)

        # Prepare response
        response = {
            "status": "success",
            "filtered_data": filtered_results,
            "total_matches": len(filtered_results)
        }

        return response

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format in area coordinates"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


class DataModel(BaseModel):
    new_data: dict
    file_name: str


@app.post("/save_data/")
async def append_to_json_file(data: DataModel):
    try:
        filename = f"./tmp/annotted_{data.file_name}.json"
        # Check if file exists and has content
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            # Read existing data
            with open(filename, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []

            # Ensure existing_data is a list
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
        else:
            existing_data = []
        print("dataa", data.new_data)
        # Append new data
        full_tagged_data = {
            "auto_tagged_data": existing_data,
            "human_tagged_data": data.new_data
        }

        datee = datetime.datetime.now().strftime("%d-%m-%Y")

        database_folder_path = "./database"
        main_filename = f"{database_folder_path}/{datee}_{data.file_name}.json"
        os.makedirs(database_folder_path, exist_ok=True)

        # Write back to file
        with open(main_filename, 'w') as file:
            json.dump(full_tagged_data, file, indent=4)

        return {"success": True, "message": "Data saved successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

