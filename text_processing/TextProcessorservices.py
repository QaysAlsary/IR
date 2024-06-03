from fastapi import Request, FastAPI, HTTPException,Request
from typing import Optional
import json
import os
from urllib.parse import urlparse
from TextProcessor import TextProcessor
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()
text_processor = TextProcessor()



@app.post("/process-file/")
async def process_file(request: Request):
    try:
        # Extract JSON data from the request
        request_body = await request.json()
        input_file_url = Path.cwd() / request_body.get("input_file_url")
        output_file_url = Path.cwd() / request_body.get("output_file_url")
        dataset_type = request_body.get("dataset_type",)
        # Validate dataset_type
        if dataset_type not in ["generic", "specific"]:
            raise HTTPException(status_code=400, detail="Invalid dataset type.")
        
        processed_data = text_processor.process_json_data(input_file_url, dataset_type, output_file_url)
        # Optionally, return processed data in the response
        return "done processeing"
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_text/")
async def process_text(request: Request):
    request_body = await request.json()
    query = request_body.get("query")
    dataset_type = request_body.get("dataset_type", "generic")  # Default to "generic" if not provided
        
    # Validate dataset_type
    if dataset_type not in ["generic", "specific"]:
        raise HTTPException(status_code=400, detail="Invalid dataset type.")
    
    # Process the query based on the dataset_type
    if dataset_type == "generic":
        processed_tokens = text_processor.process_generic(query)
    elif dataset_type == "specific":
        processed_tokens = text_processor.process_specific(query)
    
    return {"processed_tokens": processed_tokens}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
