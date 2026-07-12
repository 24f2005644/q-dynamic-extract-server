import os
import datetime
import json
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

# Load credentials from .env file if present
load_dotenv()

app = FastAPI(
    title="DataBridge Dynamic Extraction API",
    description="Extracts structured data from raw text using dynamic schemas at runtime.",
    version="1.0.0"
)

# Enable CORS (Rule 4)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported Types mapping to Python types for local/final model validation
TYPE_MAPPING = {
    "string": Optional[str],
    "integer": Optional[int],
    "float": Optional[float],
    "boolean": Optional[bool],
    "date": Optional[datetime.date],
    "array[string]": Optional[List[str]],
    "array[integer]": Optional[List[int]],
}

class ExtractionRequest(BaseModel):
    text: str
    schema_definition: Dict[str, str] = Field(..., alias="schema")

def create_dynamic_pydantic_model(schema_dict: Dict[str, str]):
    """
    Dynamically creates a Pydantic model at runtime based on the provided schema.
    All fields are made Optional with a default of None to support returning null for missing fields.
    """
    fields = {}
    for field_name, type_str in schema_dict.items():
        clean_type_str = type_str.lower().strip()
        py_type = TYPE_MAPPING.get(clean_type_str, Optional[Any])
        fields[field_name] = (py_type, None)
    return create_model("DynamicExtractionModel", **fields)

def coerce(value: Any, typ: str) -> Any:
    """
    Force the LLM output to the exact JSON type the schema asked for.
    Handles: string, integer, float, boolean, date, array[string], array[integer].
    """
    if value is None:
        return None
    try:
        t = str(typ).lower().strip()
        if t == "integer":
            return int(round(float(str(value).replace(",", ""))))
        if t in ("float", "number"):
            return float(str(value).replace(",", ""))
        if t == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "1", "yes", "y")
        if t == "date":
            # Strip whitespace, format as string or let Pydantic handle it
            return str(value).strip()
        if t == "array[integer]":
            lst = value if isinstance(value, list) else [value]
            return [int(round(float(x))) for x in lst]
        if t.startswith("array"):  # array[string] / array
            lst = value if isinstance(value, list) else [value]
            return [str(x).strip().rstrip(".").strip() if isinstance(x, str) else x for x in lst]
        # plain string: trim and drop a trailing sentence period ("Alpha Store." -> "Alpha Store")
        return str(value).strip().rstrip(".").strip()
    except Exception:
        return None

def parse_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

def extract_with_aipipe(text: str, schema_dict: Dict[str, str], token: str) -> Dict[str, Any]:
    """
    Extracts structured data using AIPipe.org proxy endpoint.
    """
    from openai import OpenAI
    
    # AIPipe exposes an OpenAI-compatible endpoint
    client = OpenAI(
        base_url="https://aipipe.org/openai/v1",
        api_key=token
    )
    
    prompt = (
        "Extract variables from the text. Return JSON with EXACTLY these keys:\n"
        f"{json.dumps(schema_dict, indent=2)}\n\n"
        "Rules: dates -> ISO YYYY-MM-DD; integer/float -> JSON numbers (not strings); "
        "boolean -> true/false; array[...] -> JSON array; if a field cannot be found use null. "
        "Extract the SHORTEST exact value (e.g. for a name give just the name).\n\n"
        f"TEXT:\n{text}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    raw_content = response.choices[0].message.content or "{}"
    return parse_json(raw_content)

def extract_with_gemini(text: str, dynamic_model: Any, token: str) -> Dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=token)
    prompt = f"Extract matching fields from the following text:\n\n{text}"
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=dynamic_model,
            temperature=0.0,
        ),
    )
    return json.loads(response.text)

def extract_with_openai(text: str, dynamic_model: Any, token: str) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=token)
    prompt = f"Extract structured data from this text:\n\n{text}"
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise data extraction agent. Extract only the fields defined in the schema. If a field cannot be found, output null for it."},
            {"role": "user", "content": prompt}
        ],
        response_format=dynamic_model,
        temperature=0.0,
    )
    
    parsed_model = response.choices[0].message.parsed
    if parsed_model:
        return parsed_model.model_dump()
    raise ValueError("OpenAI extraction returned empty result.")

@app.post("/dynamic-extract")
async def dynamic_extract(request: ExtractionRequest):
    text = request.text
    schema_dict = request.schema_definition
    keys = list(schema_dict.keys())

    aipipe_token = os.environ.get("AIPIPE_TOKEN")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # 1. Select LLM provider and fetch result
    raw_result = None
    if aipipe_token:
        try:
            raw_result = extract_with_aipipe(text, schema_dict, aipipe_token)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"AIPipe extraction failed: {str(e)}")
    elif gemini_key:
        try:
            DynamicModel = create_dynamic_pydantic_model(schema_dict)
            raw_result = extract_with_gemini(text, DynamicModel, gemini_key)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini extraction failed: {str(e)}")
    elif openai_key:
        try:
            DynamicModel = create_dynamic_pydantic_model(schema_dict)
            raw_result = extract_with_openai(text, DynamicModel, openai_key)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI extraction failed: {str(e)}")
    else:
        raise HTTPException(
            status_code=501,
            detail="No API credentials configured. Please set AIPIPE_TOKEN, GEMINI_API_KEY, or OPENAI_API_KEY in the environment."
        )

    # 2. Strict type validation and coercion
    try:
        # First pass: Apply robust type-coercion function to each key
        coerced_result = {k: coerce(raw_result.get(k, None), schema_dict[k]) for k in keys}
        
        # Second pass: Run through a dynamic Pydantic model to guarantee schema validation
        DynamicModel = create_dynamic_pydantic_model(schema_dict)
        validated_instance = DynamicModel(**coerced_result)
        
        # Final serialization to JSON values
        response_json = validated_instance.model_dump(mode="json")
        
        # Ensure we return exactly the requested keys in the original order
        return {key: response_json.get(key, None) for key in keys}

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Extraction succeeded, but validation failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "aipipe_configured": bool(os.environ.get("AIPIPE_TOKEN")),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
    }
