from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict
from pydantic import BaseModel
import json
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
import os
import ast
import datetime
from fastapi.middleware.cors import CORSMiddleware
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
import openai

from openai import OpenAI

client = openai.OpenAI()

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


def gpt_tagger(ocr_text, fields):
    
    prompt5 = """You are an advanced Named Entity Recognition (NER) model designed to tag entities from clinical OCR-extracted data, using both text and spatial coordinates for accurate medical context. Your task is to identify and label clinical text entries with the most relevant entity tags, considering both textual content and spatial positioning.

        1. Goal: Identify and label the following clinical entities EXACTLY as listed - no modifications to entity names are allowed:
        - %s

        2. Output Requirements:
        - Provide ONLY the output tuple with no additional text, comments, markdown formatting, or explanations
        - Do not include any ``` markers or other formatting
        - Start directly with the opening parenthesis ( and end with the closing parenthesis )
        - Each entry must be in the exact format: (id, 'text', 'entity_tag')
        - Use EXACTLY the entity tags as provided in the goal list - do not modify, abbreviate, or create new tags
        - Maintain exact case sensitivity and spacing in entity tags
        - If an entity doesn't match any specified tags, use exactly '0' as the tag

        3. Clinical Header and Value Processing Rules:
        - Clinical section headers (e.g., "Chief Complaint:", "History:", "Medications:") should always be tagged as '0'
        - The actual clinical content below headers should receive the appropriate entity tags
        - Headers serve as context indicators for tagging the medical information that follows them
        - Do not include header text as part of the tagged entities

        4. Spatial and Medical Contextual Analysis Rules:
        a. Vertical Relationships:
        - Items with similar x-coordinates but different y-coordinates are likely related (e.g., medication name and dosage)
        - Section headers typically appear above their corresponding clinical content
        - Example: If "Medications:" appears at coordinates [150, 200], related drug information should have similar x-coordinates

        b. Horizontal Relationships:
        - Items with similar y-coordinates are likely part of the same clinical entry
        - Items in close horizontal proximity might form compound medical entities (e.g., drug name + strength)

        c. Clinical Section Analysis:
        - Medical document sections are typically separated by significant vertical spacing
        - Related clinical items within a section usually have similar x-coordinates or consistent indentation
        - Section headers (History, Physical Exam, Assessment, Plan) usually appear at the top of their respective groups

        5. Multi-Line Clinical Entity Rules:
        - Group entries as a single entity if they:
            * Share similar x-coordinates (within a ±20 unit margin)
            * Have sequential y-coordinates (within a reasonable vertical spacing)
            * Form a logical medical unit (e.g., complete diagnosis, medication with dosage, multi-line symptoms)
        - Each component of a multi-line clinical entity should receive the exact same entity tag
        - Consider relative spacing between lines to determine medical grouping

        6. Input Data Structure: Each input item contains:
        (id, 'text', [x, y])
        where:
        - x, y: center coordinates of the text
        - Lower y-values indicate higher position on the page
        - x-values represent left-to-right positioning

        7. Coordinate-Based Clinical Analysis:
        - Verify spatial relationships between medical elements:
            * Clinical headers and their corresponding content typically share x-coordinates
            * Related medical items usually have y-coordinates within 50 units
            * Items in the same clinical column have x-coordinates within ±20 units
        - Use coordinate patterns to distinguish between similar medical fields in different sections
        - Consider standard medical document layout patterns for entity identification
        - Pay attention to typical clinical note structures (SOAP format, discharge summaries, etc.)

        8. Medical Context Considerations:
        - Consider medical abbreviations and their spatial context
        - Account for typical clinical documentation patterns
        - Recognize standard medical formatting (vital signs, lab values, medications)
        - Differentiate between patient information and clinical findings based on positioning

        Example Input:
        (1, 'Chief Complaint:', [150.0, 200.0]),
        (2, 'Chest pain', [150.0, 220.0]),
        (3, 'Medications:', [150.0, 280.0]),
        (4, 'Aspirin 81mg daily', [150.0, 300.0])

        Example Output:
        (
            (1, 'Chief Complaint:', '0'),
            (2, 'Chest pain', 'Symptom'),
            (3, 'Medications:', '0'),
            (4, 'Aspirin 81mg daily', 'Medication')
        )

        IMPORTANT:
        - Return ONLY the output tuple without any additional text or formatting
        - Use EXACTLY the entity tags as provided - no modifications allowed
        - Default to '0' for non-matching entities
        - Always tag clinical headers as '0' and apply entity tags to the actual medical content below them
        - Consider medical context and standard clinical documentation patterns
        """ % (fields)
    
    prompt5 = """You are an advanced Named Entity Recognition (NER) model designed to tag entities from clinical OCR-extracted data, using both text and spatial coordinates for accurate medical context. Your task is to identify and label clinical text entries with the most relevant entity tags, considering both textual content and spatial positioning.

        1. Goal: Automatically identify and create entity tags for clinical entities found in the document. Create entity tags in the following format:
        - Disease entities: Use format "full_disease_name_X" where X is a sequential number (1, 2, 3, etc.)
          Examples: "full_disease_name_1", "full_disease_name_2", "full_disease_name_3"
        - Drug/Medication entities: Use format "drug_X" where X is a sequential number (1, 2, 3, etc.)
          Examples: "drug_1", "drug_2", "drug_3"
        - The number of disease and drug entities can vary from 0 to 10 or more depending on the document content
        - Create new entity tags dynamically based on what you find in the clinical text

        2. Disease Entity Identification Rules:
        - Include the COMPLETE disease name including all descriptive qualifiers and condition modifiers
        - Include severity indicators: mild, moderate, severe, acute, chronic
        - Include type specifications: primary, secondary, essential, idiopathic
        - Include certainty modifiers: unspecified, specified, without complications, with complications
        - Include stage/grade information: stage 1, stage 2, grade I, grade II, etc.
        - Include laterality: left, right, bilateral
        - Include anatomical locations when part of the disease name
        
        Examples of complete disease names to capture:
        - "Type 2 diabetes mellitus without complications"
        - "Essential (primary) hypertension" 
        - "Chronic kidney disease, stage 2 (mild)"
        - "Hyperlipidemia, unspecified"
        - "Acute myocardial infarction, anterior wall"
        - "Chronic obstructive pulmonary disease, moderate"

        3. Entity Tag Creation Rules:
        - For each unique complete disease/condition found (including all modifiers), create a new "full_disease_name_X" tag
        - For each unique medication/drug found, create a new "drug_X" tag
        - Assign the same tag to multiple instances of the identical complete disease name
        - Start numbering from 1 for each entity type
        - Only create tags for entities that actually appear in the document
        - Treat diseases with different modifiers as separate entities (e.g., "mild hypertension" vs "severe hypertension")

        4. Output Requirements:
        - Provide ONLY the output tuple with no additional text, comments, markdown formatting, or explanations
        - Do not include any ``` markers or other formatting
        - Start directly with the opening parenthesis ( and end with the closing parenthesis )
        - Each entry must be in the exact format: (id, 'text', 'entity_tag')
        - Use the dynamically created entity tags as described above
        - If an entity doesn't match any medical category, use exactly '0' as the tag

        5. Clinical Header and Value Processing Rules:
        - Clinical section headers (e.g., "Chief Complaint:", "History:", "Medications:", "Diagnosis:") should always be tagged as '0'
        - The actual clinical content below headers should receive the appropriate entity tags
        - Headers serve as context indicators for tagging the medical information that follows them
        - Do not include header text as part of the tagged entities

        6. Spatial and Medical Contextual Analysis Rules:
        a. Vertical Relationships:
        - Items with similar x-coordinates but different y-coordinates are likely related (e.g., medication name and dosage)
        - Section headers typically appear above their corresponding clinical content
        - Example: If "Medications:" appears at coordinates [150, 200], related drug information should have similar x-coordinates

        b. Horizontal Relationships:
        - Items with similar y-coordinates are likely part of the same clinical entry
        - Items in close horizontal proximity might form compound medical entities (e.g., drug name + strength)

        c. Clinical Section Analysis:
        - Medical document sections are typically separated by significant vertical spacing
        - Related clinical items within a section usually have similar x-coordinates or consistent indentation
        - Section headers (History, Physical Exam, Assessment, Plan) usually appear at the top of their respective groups

        7. Multi-Line Clinical Entity Rules:
        - Group entries as a single entity if they:
            * Share similar x-coordinates (within a ±20 unit margin)
            * Have sequential y-coordinates (within a reasonable vertical spacing)
            * Form a logical medical unit (e.g., complete diagnosis with modifiers, medication with dosage)
        - Each component of a multi-line clinical entity should receive the exact same entity tag
        - Consider relative spacing between lines to determine medical grouping
        - Pay special attention to disease names that span multiple lines with their qualifiers

        8. Input Data Structure: Each input item contains:
        (id, 'text', [x, y])
        where:
        - x, y: center coordinates of the text
        - Lower y-values indicate higher position on the page
        - x-values represent left-to-right positioning

        9. Coordinate-Based Clinical Analysis:
        - Verify spatial relationships between medical elements:
            * Clinical headers and their corresponding content typically share x-coordinates
            * Related medical items usually have y-coordinates within 50 units
            * Items in the same clinical column have x-coordinates within ±20 units
        - Use coordinate patterns to distinguish between similar medical fields in different sections
        - Consider standard medical document layout patterns for entity identification
        - Pay attention to typical clinical note structures (SOAP format, discharge summaries, etc.)

        10. Medical Context Considerations:
        - Consider medical abbreviations and their spatial context
        - Account for typical clinical documentation patterns
        - Recognize standard medical formatting (vital signs, lab values, medications)
        - Differentiate between patient information and clinical findings based on positioning
        - Identify complete diseases/conditions including all descriptive modifiers
        - Identify drugs/medications including: prescription drugs, over-the-counter medications, dosages, administration routes

        11. Entity Assignment Logic:
        - Scan the entire document first to identify all unique complete diseases (with modifiers) and drugs
        - Assign sequential numbers to each unique entity type
        - Apply consistent tagging throughout the document for the same complete entities
        - Remember that the same base disease with different modifiers should get different entity numbers

        Example Input:
        (1, 'Diagnosis:', [150.0, 200.0]),
        (2, 'Type 2 diabetes mellitus', [150.0, 220.0]),
        (3, 'without complications', [150.0, 240.0]),
        (4, 'Essential (primary)', [150.0, 280.0]),
        (5, 'hypertension', [150.0, 300.0]),
        (6, 'Medications:', [150.0, 340.0]),
        (7, 'Metformin 500mg', [150.0, 360.0])

        Example Output:
        (
            (1, 'Diagnosis:', '0'),
            (2, 'Type 2 diabetes mellitus', 'full_disease_name_1'),
            (3, 'without complications', 'full_disease_name_1'),
            (4, 'Essential (primary)', 'full_disease_name_2'),
            (5, 'hypertension', 'full_disease_name_2'),
            (6, 'Medications:', '0'),
            (7, 'Metformin 500mg', 'drug_1')
        )

        IMPORTANT:
        - Return ONLY the output tuple without any additional text or formatting
        - Create entity tags dynamically based on the clinical content found
        - Include ALL descriptive modifiers as part of the complete disease name
        - Use consistent numbering for the same complete diseases and drugs throughout the document
        - Default to '0' for non-medical entities
        - Always tag clinical headers as '0' and apply entity tags to the actual medical content below them
        - Consider medical context and standard clinical documentation patterns
        - Treat diseases with different qualifiers as separate entities
        """

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt5},
            {"role": "user", "content": f"Provided Input:{ocr_text}"}
        ]
    )
    gpt_responce = completion.choices[0].message.content
    print(gpt_responce)
    try:
        x = ast.literal_eval(gpt_responce)
        x = list(x)
        print(x)
        return x
    except Exception as e:
        print(e)
        return None

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

    # Simulate GPT tagging
    gpt_response = None
    if modee == 1:
        while gpt_response is None:
            gpt_response = gpt_tagger(text_content, fields)

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
            label = gpt_response[i - 1][2]
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
            modee = int(json_dict["data"]["mode"])  # mode = 1 for gpt tagging
            # modee=int(input("1 for gpt tagging and 0 for manual tagging:"))
        except:
            modee = 1
            print("defaulting to gpt tagging")

    except json.JSONDecodeError:
        return JSONResponse(content={"message": "Invalid JSON data"}, status_code=400)

    # Step 3: Process the image with OCR and extract fields
    output_img_path = f"./tmp/annotted_{file.filename}"
    try:
        data = process_image_with_ocr(temp_file_path, output_img_path, fields, modee)  # mode=True for gpt tagging
    except Exception as e:
        return JSONResponse(content={"message": f"OCR processing failed: {str(e)}"}, status_code=500)

    # Step 4: Prepare the response
    response = {
        "fields": fields,
        "data": data,
    }
    print("gptttttttttttttttt---", response)

    #cleaned_response = clean_entity_values(response, response["fields"])

    return JSONResponse(content=response)


from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
            "gpt_tagged_data": existing_data,
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

