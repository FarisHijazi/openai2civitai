"""
FastAPI proxy server to translate between OpenAI Images API and CivitAI API.

This application exposes OpenAI-compatible image generation endpoints and translates
requests to the CivitAI API backend, returning responses in OpenAI format.
"""

# TODO: add support for adding model URLs and that auto-converts to URNs
import asyncio
import base64
import warnings
import backoff
import copy
import json
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional
from multiprocessing.pool import ThreadPool


import civitai
import httpx
import joblib
import requests
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import prompt_reviser

"""
Configuration module for the OpenAI to CivitAI Image API Proxy.

This module handles environment variable loading and default settings.
"""


# Server Configuration
POLL_TIMEOUT: float = float(os.getenv("POLL_TIMEOUT", "600"))
POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", "2"))


memory = joblib.Memory(location="./cache", verbose=0)
# official SDXL
CIVITAI_DEFAULT_MODEL = os.getenv(
    "CIVITAI_DEFAULT_MODEL", "urn:air:sdxl:checkpoint:civitai:101055@128078"
)
CIVITAI_DEFAULT_NEGATIVEPROMPT = os.getenv(
    "CIVITAI_DEFAULT_NEGATIVEPROMPT",
    "easynegative, badhandv4, (worst quality, low quality, normal quality), bad-artist, blurry, ugly, ((bad anatomy)),((bad hands)),((bad proportions)),((duplicate limbs)),((fused limbs)),((interlocking fingers)),((poorly drawn face)), signature, watermark, artist logo, patreon logo",
)

IMAGE_PROMPT_TEMPLATE = os.getenv("IMAGE_PROMPT_TEMPLATE", "{prompt}")
PROMPT_REVISER_TRIGGER_PREFIX = "REV//"

GENERATION_PARAMS_DEFAULTS = {
    "negativePrompt": CIVITAI_DEFAULT_NEGATIVEPROMPT,
    "scheduler": "EulerA",
    "steps": 25,
    "cfgScale": 4,
    "width": 1024,
    "height": 1024,
    "clipSkip": 2,
    "seed": None,
}


class OpenAIImageRequest(BaseModel):
    """OpenAI-compatible image generation request model."""

    prompt: str = Field(
        ..., description="Text description of the desired image", max_length=4000
    )
    model: Optional[str] = Field(
        default=CIVITAI_DEFAULT_MODEL, description="Model to use for image generation"
    )
    n: Optional[int] = Field(
        default=1, ge=1, le=10, description="Number of images to generate"
    )
    quality: Optional[Literal["standard", "hd"]] = Field(
        default=None, description="Image quality"
    )
    response_format: Optional[Literal["url", "b64_json"]] = Field(
        default="url", description="Response format"
    )
    size: Optional[
        Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792", "auto"]
    ] = Field(default="auto", description="Size of generated images")
    style: Optional[Literal["vivid", "natural"]] = Field(
        default=None, description="Image style"
    )
    user: Optional[str] = Field(
        default=None, description="Unique identifier for the user"
    )
    seed: Optional[int] = Field(default=None, description="Seed for image generation")


class OpenAIImageData(BaseModel):
    """Single image data in OpenAI response format."""

    url: Optional[str] = Field(default=None, description="URL to the generated image")
    b64_json: Optional[str] = Field(
        default=None, description="Base64 encoded image data"
    )
    revised_prompt: Optional[str] = Field(
        default=None, description="Revised prompt used for generation"
    )


class OpenAIImageResponse(BaseModel):
    """OpenAI-compatible image generation response model."""

    created: int = Field(
        ..., description="Unix timestamp of when the image was created"
    )
    data: List[OpenAIImageData] = Field(..., description="List of generated images")


class OpenAIErrorResponse(BaseModel):
    """OpenAI-compatible error response model."""

    error: dict = Field(..., description="Error details")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    try:
        if prompt_reviser.is_available():
            print("Prompt reviser is available")
        else:
            print(
                "Prompt reviser is not available, enable prompt revision by setting the following environment variables: PROMPT_REVISER_OPENAI_API_KEY, PROMPT_REVISER_OPENAI_BASE_URL, PROMPT_REVISER_MODEL"
            )
        assert os.environ[
            "CIVITAI_API_TOKEN"
        ], "CIVITAI_API_TOKEN environment variable is required. Please set it to your CivitAI API token."
        print(f"✅ OpenAI to CivitAI Image API Proxy started successfully")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        raise
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise

    yield

    # Shutdown
    print("🛑 OpenAI to CivitAI Image API Proxy shutting down")


app = FastAPI(
    title="OpenAI to CivitAI Image API Proxy",
    description="A proxy server that translates between OpenAI Images API and CivitAI API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "openai-civitai-proxy",
        "prompt_reviser available": prompt_reviser.is_available(),
    }


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Request validation failed",
                "type": "invalid_request_error",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions in OpenAI-compatible format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": (
                    "api_error" if exc.status_code >= 500 else "invalid_request_error"
                ),
                "code": None,
            }
        },
    )


async def fetch_image_as_base64(url: str) -> str:
    """
    Fetches an image from a URL and returns it as a base64 encoded string.

    Args:
        url: The URL of the image to fetch.

    Returns:
        The base64 encoded image content.

    Raises:
        HTTPException: If the image cannot be fetched.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
    except httpx.HTTPStatusError as e:
        print(f"Error fetching image: {e.response.status_code} from {url}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch image from provider: {e.response.status_code}",
        )
    except Exception as e:
        print(f"Unexpected error fetching image: {e}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error fetching image: {str(e)}"
        )


@memory.cache
def urn_str2dict(urn: str) -> dict:
    """

    Args:
        urn: A URN string that may contain a strength modifier (e.g. "urn:air:sdxl:lora:civitai:1115064@1253021*0.5")

    Returns:
        dict: A dictionary containing the URN, type and strength information

    Example:
        >>> urn_str2dict("urn:air:sdxl:lora:civitai:1115064@1253021*0.5")
        {'urn:air:sdxl:lora:civitai:1115064@1253021': {'type': 'Lora', 'strength': 0.5}}

        $ urn_str2dict("urn:air:sdxl:lora:civitai:1115064@1253021*0.5")
        >>> {"urn:air:sdxl:lora:civitai:1115064@1253021": {"type": "Lora", "strength": 0.5}}

        $ urn_str2dict("urn:air:sdxl:embedding:civitai:1115064@1253021!easynegative")
        >>> {"urn:air:sdxl:embedding:civitai:1115064@1253021": {"type": "TextualInversion", "strength": 1.0, "triggerWord": "easynegative"}}

        $ urn_str2dict("urn:air:sdxl:embedding:civitai:1115064@1253021*1.7!easynegative")
        >>> {"urn:air:sdxl:embedding:civitai:1115064@1253021": {"type": "TextualInversion", "strength": 1.7, "triggerWord": "easynegative"}}

        # in the cas of a url, we need to convert it to a urn using the civitai_url_to_urn function
        $ urn_str2dict("https://civitai.com/models/341353?modelVersionId=382152")
        >>> {"urn:air:sdxl:lora:civitai:341353@382152": {"type": "Lora", "strength": 1.0}}

    """
    additionalNetworks_types_map = {
        "TextualInversion": "embedding",
        "Lora": "lora",
    }

    if not urn:
        return {}

    if urn.startswith("https://civitai.com/models/"):
        urn = civitai_url_to_urn(urn)

    trigger_word = None
    if "!" in urn:
        urn, trigger_word = urn.strip().split("!", 1)
        trigger_word = trigger_word.strip()

    # Split URN and strength if present
    strength = 1.0
    if "*" in urn:
        urn_part, strength = urn.split("*")
        strength = float(strength.strip().strip(","))
    else:
        urn_part = urn

    # Determine type from URN
    network_type = urn_part.split(":")[3]  # Get 'lora' or 'embedding' from URN
    type_name = next(
        (k for k, v in additionalNetworks_types_map.items() if v == network_type),
        "TextualInversion",  # Default to TextualInversion if not found
    )

    result = {
        urn_part: {
            "type": type_name,
            "strength": strength,
        }
    }
    if trigger_word:
        result[urn_part]["triggerWord"] = trigger_word
    return result


@memory.cache
def civitai_url_to_urn(url: str) -> str:
    model_id = url.split("/models/")[1].split("/")[0]
    model_version_id = (
        url.split("modelVersionId=")[1].split("&")[0]
        if "modelVersionId=" in url
        else None
    )
    response = requests.get(f"https://civitai.com/api/v1/models/{model_id}")
    response.raise_for_status()
    model_data = response.json()
    model_type = model_data["type"]
    model_version_id = model_data["modelVersionId"]
    urn = f"urn:air:{model_type}:civitai:{model_id}@{model_version_id}"
    return urn


def civitai_urn_to_url(urn: str) -> str:
    """
    Converts a CivitAI URN to a web URL.

    Args:
        urn: CivitAI URN in format like "urn:air:sdxl:lora:civitai:212532@239420"

    Returns:
        str: The corresponding CivitAI web URL

    Example:
        >>> civitai_urn_to_url("urn:air:sdxl:lora:civitai:212532@239420")
        "https://civitai.com/models/212532?modelVersionId=239420"
    """
    if not urn.startswith("urn:air:"):
        raise ValueError(f"Invalid URN format: {urn}")

    # Split URN parts: urn:air:sdxl:lora:civitai:212532@239420
    parts = urn.split(":")
    if len(parts) < 6 or parts[4] != "civitai":
        raise ValueError(f"Invalid CivitAI URN format: {urn}")

    # Extract model ID and version ID from the last part
    model_part = parts[5]  # "212532@239420"
    if "@" not in model_part:
        raise ValueError(f"Invalid model part in URN: {model_part}")

    model_id, version_id = model_part.split("@", 1)

    return f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"


def parse_prompt_components(user_input):
    user_input = user_input.strip()
    other_metadata_possible_keys = [
        "Steps:",
        "CFG scale:",
        "Sampler:",
        "Seed:",
        "Size:",
        "Model:",
        "width:",
        "height:",
        "draft:",
        "baseModel:",
        "disablePoi:",
        "aspectRatio:",
        "Created Date:",
        "experimental:",
        "Clip skip:",
        "process:",
        "workflow:",
        "quantity:",
    ]

    def split_off_prompt(text):
        triggers = [
            "Negative prompt:",
            "Additional networks:",
        ] + other_metadata_possible_keys
        pattern = re.compile("|".join(map(re.escape, triggers)))
        match = pattern.search(text)
        if not match:
            return text, ""
        return text[: match.start()].rstrip(", \n"), text[match.start() :]

    def grab_section(label, text, all_triggers):
        pattern_str = "|".join(re.escape(t) for t in all_triggers if t != label)
        regex = re.compile(
            rf"{re.escape(label)}\s*(?P<content>.*?)(?=(?:{pattern_str})|$)",
            flags=re.DOTALL,
        )
        m = regex.search(text)
        if not m:
            return "", text
        content = m.group("content")
        start, end = m.span()
        new_text = text[:start] + text[end:]
        return content.strip(), new_text.strip()

    user_input = user_input.strip()
    all_triggers = [
        "Negative prompt:",
        "Additional networks:",
    ] + other_metadata_possible_keys

    prompt, remainder = split_off_prompt(user_input)
    negative_prompt, remainder = grab_section(
        "Negative prompt:", remainder, all_triggers
    )
    additional_networks, remainder = grab_section(
        "Additional networks:", remainder, all_triggers
    )
    other_metadata = remainder.strip()

    return {
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "additionalNetworks": additional_networks,
        "other_metadata": other_metadata,
    }


def parse_generation_data(input_string: str) -> Dict[str, Any]:
    """
    Parse a CivitAI prompt block into its logical components using regex.

    Args:
        input_string (str): The input prompt string containing prompt, negative prompt and metadata

    Returns:
        Dict[str, Any]: Dictionary containing:
            - prompt (str): The main prompt text
            - negativePrompt (str): The negative prompt text (empty if not found)
            - additionalNetworks (Dict[str, Any]): Dictionary of additional networks (empty if not found)
            - other_metadata (Dict[str, Any]): Dictionary of additional metadata parameters (empty if not found)
    """

    def convert_sentence_to_camel_case(sentence: str) -> str:
        words = sentence.lower().strip().split(" ")
        return words[0] + "".join(word.capitalize() for word in words[1:])

    component_strings = parse_prompt_components(input_string)

    # Parse additional networks into list then convert to dict
    additional_networks_list = []
    if component_strings["additionalNetworks"]:
        additional_networks_list = [
            item.strip()
            for item in re.split(r",\s*", component_strings["additionalNetworks"])
            if item.strip()
        ]

    additionalNetworks = {}
    for urn in additional_networks_list:
        urn_dict = urn_str2dict(urn.strip())
        additionalNetworks.update(urn_dict)

    # Merge metadata lines and break into individual key-value pairs

    # Clean up metadata text by removing Created Date which contains colons
    metadata_text = re.sub(
        r"Created Date:[^,]*,?\s*", "", component_strings["other_metadata"]
    ).strip()
    metadata = {}
    for part in re.split(r",\s*", metadata_text):
        if ":" not in part:
            continue

        key, value = part.split(":", 1)
        key = convert_sentence_to_camel_case(key.strip())
        value = value.strip().replace(" ", "")

        # Convert values to appropriate types
        if value.replace(".", "").replace("-", "").isdigit():
            metadata[key] = float(value) if "." in value else int(value)
        elif value.lower() in ("true", "false"):
            metadata[key] = value.lower() == "true"
        else:
            metadata[key] = value

    if (not metadata.get("width") or not metadata.get("height")) and metadata.get(
        "size"
    ):
        width, height = metadata["size"].split("x")
        metadata["width"] = int(width)
        metadata["height"] = int(height)

    return {
        "prompt": component_strings["prompt"],
        "negativePrompt": component_strings["negativePrompt"],
        "additionalNetworks": additionalNetworks,
        "other_metadata": metadata,
    }


def parse_inputstring_to_generation_input(input_string: str) -> dict:
    generation_data = parse_generation_data(input_string)
    print(f"generation_data={json.dumps(generation_data, indent=4)}")

    # Validate model is in URN format
    model = generation_data["other_metadata"].get("model", CIVITAI_DEFAULT_MODEL)
    if not model.startswith("urn:"):
        raise ValueError(
            f"Model must be in URN format (starting with 'urn:'), got: {model}"
        )

    METADATA_WHITELIST = [
        "prompt",
        "negativePrompt",
        "width",
        "height",
        "scheduler",
        "steps",
        "cfgScale",
        "quantity",
        "seed",
        "clipSkip",
    ]
    params = copy.deepcopy(GENERATION_PARAMS_DEFAULTS)
    for k, v in generation_data["other_metadata"].items():
        if not not k in METADATA_WHITELIST:
            params[k] = v
    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}
    params["prompt"] = generation_data["prompt"]
    params["negativePrompt"] = generation_data["negativePrompt"]

    generation_input = dict(
        model=generation_data["other_metadata"].get("model", CIVITAI_DEFAULT_MODEL),
        params=params,
        additionalNetworks=generation_data["additionalNetworks"],
    )

    # yield f"generation_input\n```json\n{json.dumps(generation_input, indent=4)}\n```\n"
    if generation_input["params"]["width"] > 1024:
        warning_message = f"\n\n⚠️ Warning: Width {generation_input['params']['width']} is too large. Please set it to 1024 or less. Setting to 1024.\n\n"
        warnings.warn(warning_message)
        generation_input["params"]["width"] = 1024
    if generation_input["params"]["height"] > 1024:
        warning_message = f"\n\n⚠️ Warning: Height {generation_input['params']['height']} is too large. Please set it to 1024 or less. Setting to 1024.\n\n"
        warnings.warn(warning_message)
        generation_input["params"]["height"] = 1024

    return generation_input


def translate_openai_to_civitai(request: OpenAIImageRequest, i: int = 0) -> dict:
    """
    Translate OpenAI image request format to CivitAI format.

    Args:
        request: OpenAI-formatted image generation request

    Returns:
        Dictionary formatted for CivitAI API
    """
    generation_input = parse_inputstring_to_generation_input(request.prompt)

    if request.seed:
        generation_input["params"]["seed"] = request.seed
    if request.n:
        generation_input["params"]["quantity"] = request.n
    if request.size and request.size != "auto":
        width, height = request.size.split("x")
        generation_input["params"]["width"] = int(width)
        generation_input["params"]["height"] = int(height)

    if "seed" in generation_input["params"]:
        generation_input["params"]["seed"] += i

    generation_input["params"]["prompt"] = IMAGE_PROMPT_TEMPLATE.format(
        prompt=generation_input["params"]["prompt"].strip()
    ).strip()

    if generation_input["params"]["prompt"].startswith(PROMPT_REVISER_TRIGGER_PREFIX):
        generation_input["params"]["prompt"] = generation_input["params"]["prompt"].lstrip(PROMPT_REVISER_TRIGGER_PREFIX).strip()
        print(f"Prompt revision triggered by user")

        if not prompt_reviser.is_available():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt revision is disabled. Please enable it by setting the following environment variables: PROMPT_REVISER_OPENAI_API_KEY, PROMPT_REVISER_OPENAI_BASE_URL, PROMPT_REVISER_MODEL",
            )

        generation_input["params"]["prompt"] = (
            generation_input["params"]["prompt"]
            .replace("\n", " ")
            .replace("  ", " ")
            + " " * i
        )
        revised_prompt = prompt_reviser.revise_prompt(
            generation_input["params"]["prompt"]
        )
        generation_input["params"]["prompt"] = revised_prompt.replace(
            "\n", " "
        ).replace("  ", " ")
        print(
            f"Revised prompt \"{generation_input['params']['prompt']}\" -> \"{revised_prompt}\""
        )


    # Map quality to CivitAI parameters
    if request.quality:
        # Higher quality = more steps and lower CFG scale for better results
        if request.quality == "hd":
            generation_input["params"]["cfgScale"] = 4.0
            generation_input["params"]["steps"] = 30
        else:  # "standard"
            generation_input["params"]["cfgScale"] = 7.0
            generation_input["params"]["steps"] = 20

    # Use the configured default CivitAI model
    if request.style:
        # Adjust negative prompt based on style
        if request.style == "natural":
            generation_input["params"][
                "prompt"
            ] += ", natural lighting, soft colors, realistic style, subtle details"
        else:  # "vivid"
            generation_input["params"][
                "prompt"
            ] += ", vibrant colors, high contrast, dramatic lighting, bold details, saturated"

    return generation_input


async def translate_civitai_to_openai(
    civitai_response: dict, request: OpenAIImageRequest, revised_prompt: str
) -> OpenAIImageResponse:
    """
    Translate CivitAI response format to OpenAI format.

    Args:
        civitai_response: Response from CivitAI API
        request: Original OpenAI request for context

    Returns:
        OpenAI-formatted response

    Raises:
        ValueError: If response format is invalid
    """
    try:
        job = civitai_response["jobs"][0]

        if not job.get("result") or len(job["result"]) == 0:
            raise ValueError("No results found in CivitAI response")

        result = job["result"][0]

        if not result.get("available"):
            raise ValueError("Generated image is not available")

        # Get the image URL
        image_url = result.get("blobUrl")
        if not image_url:
            raise ValueError("No image URL found in result")

        # Create image data based on requested format
        image_url_for_response = None
        b64_json_for_response = None

        if request.response_format == "url":
            image_url_for_response = image_url
        elif request.response_format == "b64_json":
            try:
                b64_json_for_response = await fetch_image_as_base64(image_url)
            except HTTPException:
                raise  # Re-raise to be caught by the main handler

        image_data = OpenAIImageData(
            url=image_url_for_response,
            b64_json=b64_json_for_response,
            revised_prompt=(
                revised_prompt
                if prompt_reviser.is_available()
                else "<Prompt revision is disabled, please enable it by setting the following environment variables: PROMPT_REVISER_OPENAI_API_KEY, PROMPT_REVISER_OPENAI_BASE_URL, PROMPT_REVISER_MODEL>"
            ),
        )

        # Use job creation time if available, otherwise current time
        created_time = job.get("createdAt", int(time.time()))
        if isinstance(created_time, str):
            # If timestamp is a string, try to parse it
            import datetime

            try:
                dt = datetime.datetime.fromisoformat(
                    created_time.replace("Z", "+00:00")
                )
                created_time = int(dt.timestamp())
            except:
                created_time = int(time.time())

        return OpenAIImageResponse(created=created_time, data=[image_data])

    except KeyError as e:
        raise ValueError(f"Missing required field in CivitAI response: {e}")
    except Exception as e:
        raise ValueError(f"Failed to parse CivitAI response: {e}")


async def poll_civitai_job(token: str, job_id: str) -> dict:
    """
    Poll CivitAI job until completion.

    Args:
        token: Job token from CivitAI
        job_id: Job ID to poll

    Returns:
        Final job result

    Raises:
        HTTPException: If job fails or times out
    """
    print(f"🔄 Starting to poll job {job_id} (max {POLL_TIMEOUT} seconds)")

    start_time = time.time()
    attempt = 0

    while (time.time() - start_time) < POLL_TIMEOUT:
        attempt += 1
        try:
            response = await civitai.jobs.get(token=token, job_id=job_id)

            if "jobs" not in response or len(response["jobs"]) == 0:
                print(f"⚠️ No jobs found in response on attempt {attempt}")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            job = response["jobs"][0]
            job_status = job.get("status", "unknown")

            elapsed_time = time.time() - start_time
            print(
                f"📊 Attempt {attempt} ({elapsed_time:.1f}s elapsed): Job status = {job_status}"
            )

            # Check if job failed
            if job_status in ["Failed", "Cancelled"]:
                error_msg = job.get(
                    "error", "Job failed without specific error message"
                )
                print(f"❌ Job failed with status {job_status}: {error_msg}")
                raise HTTPException(
                    status_code=500, detail=f"CivitAI job failed: {error_msg}"
                )

            # Check if job is complete
            if job.get("result") and len(job["result"]) > 0:
                result = job["result"][0]
                if result.get("available"):
                    print(
                        f"✅ Job completed successfully after {attempt} attempts ({elapsed_time:.1f}s)"
                    )
                    return response
                # else:
                #     print(
                #         f"⏳ Job has result but image not yet available (attempt {attempt})"
                #     )
            else:
                print(
                    f"⏳ Job still processing (attempt {attempt}), time elapsed: {elapsed_time:.1f}s/{POLL_TIMEOUT}s"
                )

            # Wait before next poll
            await asyncio.sleep(POLL_INTERVAL)

        except HTTPException:
            # Re-raise HTTP exceptions (like job failures)
            raise
        except Exception as e:
            print(f"⚠️ Error polling job {job_id} on attempt {attempt}: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    print(f"⏰ Job {job_id} timed out after {POLL_TIMEOUT} seconds")
    raise HTTPException(
        status_code=408,
        detail=f"Job timed out after {POLL_TIMEOUT} seconds. Try again or use a simpler prompt.",
    )


# @memory.cache
@backoff.on_exception(backoff.expo, Exception, max_tries=5, jitter=backoff.full_jitter)
async def civitai_image_create(civitai_input: dict, wait: bool = False):
    return await civitai.image.create(civitai_input, wait=wait)


@app.post("/v1/images/generations", response_model=OpenAIImageResponse)
async def create_image(
    request: OpenAIImageRequest,
    authorization: Optional[str] = Header(
        None, description="Bearer token for authentication"
    ),
) -> OpenAIImageResponse:
    """
    Create an image using OpenAI-compatible API that proxies to CivitAI.

    Args:
        request: Image generation request in OpenAI format
        authorization: Optional authorization header

    Returns:
        Generated image response in OpenAI format

    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Validate API token is configured
        if not os.environ["CIVITAI_API_TOKEN"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="CivitAI API token not configured",
            )

        # Validate prompt length
        if len(request.prompt.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt cannot be empty"
            )

        print(
            f"🎨 Submitting {request.n} image generation request(s): {request.size}, quality={request.quality}, style={request.style}"
        )

        # Translate request format
        with ThreadPool(request.n) as pool:
            civitai_inputs = list(
                pool.imap(
                    lambda i: translate_openai_to_civitai(request, i),
                    range(request.n or 1),
                )
            )

        with ThreadPool(request.n) as pool:
            create_tasks = list(pool.imap(civitai_image_create, civitai_inputs))

        civitai_responses = await asyncio.gather(*create_tasks)

        # Poll for completion in parallel
        polling_tasks = [
            poll_civitai_job(res["token"], res["jobs"][0]["jobId"])
            for res in civitai_responses
        ]

        final_responses = await asyncio.gather(*polling_tasks)
        revised_prompts = [
            civitai_input["params"]["prompt"] for civitai_input in civitai_inputs
        ]

        # Translate responses in parallel
        translation_tasks = [
            translate_civitai_to_openai(final_res, request, revised_prompt)
            for final_res, revised_prompt in zip(final_responses, revised_prompts)
        ]
        openai_responses = await asyncio.gather(*translation_tasks)

        # Combine all image data into single response
        all_image_data = [data for res in openai_responses for data in res.data]

        # Take the creation time from the first response, if available.
        created_time = (
            openai_responses[0].created if openai_responses else int(time.time())
        )

        return OpenAIImageResponse(created=created_time, data=all_image_data)

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation error: {str(e)}",
        )
    except Exception as e:
        print(f"Unexpected error generating image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the image",
        )


async def main():
    # testing image generation with multiple images (4)
    request = OpenAIImageRequest(
        prompt="A badass cyborg knight with 4 swords, fighting a giant monster the size of a mountain",
        n=4,
        seed=12345,
        # size="1024x1024",
        # quality="standard",
        # style="natural",
    )
    response = await create_image(request)
    urls = [data.url for data in response.data]
    import pandas as pd

    df = pd.DataFrame(
        {
            "md_image": [f"![]({url})" for url in urls],
            "response": response.data,
        }
    )
    markdown_table = df.to_markdown(index=False)

    with open("generated_images.md", "w") as f:
        f.write("# Generated Images\n\n")
        f.write(markdown_table)

    print(f"Saved {len(urls)} image URLs to generated_images.md")


if __name__ == "__main__":
    asyncio.run(main())
