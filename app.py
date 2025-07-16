"""
FastAPI proxy server to translate between OpenAI Images API and CivitAI API.

This application exposes OpenAI-compatible image generation endpoints and translates
requests to the CivitAI API backend, returning responses in OpenAI format.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import civitai
from config import config, setup_civitai
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError


class OpenAIImageRequest(BaseModel):
    """OpenAI-compatible image generation request model."""

    prompt: str = Field(..., description='Text description of the desired image', max_length=4000)
    model: Optional[str] = Field(default='dall-e-3', description='Model to use for image generation')
    n: Optional[int] = Field(default=1, ge=1, le=10, description='Number of images to generate')
    quality: Optional[Literal['standard', 'hd']] = Field(default='standard', description='Image quality')
    response_format: Optional[Literal['url', 'b64_json']] = Field(default='url', description='Response format')
    size: Optional[Literal['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']] = Field(
        default='1024x1024', description='Size of generated images'
    )
    style: Optional[Literal['vivid', 'natural']] = Field(default='vivid', description='Image style')
    user: Optional[str] = Field(default=None, description='Unique identifier for the user')


class OpenAIImageData(BaseModel):
    """Single image data in OpenAI response format."""

    url: Optional[str] = Field(default=None, description='URL to the generated image')
    b64_json: Optional[str] = Field(default=None, description='Base64 encoded image data')
    revised_prompt: Optional[str] = Field(default=None, description='Revised prompt used for generation')


class OpenAIImageResponse(BaseModel):
    """OpenAI-compatible image generation response model."""

    created: int = Field(..., description='Unix timestamp of when the image was created')
    data: List[OpenAIImageData] = Field(..., description='List of generated images')


class OpenAIErrorResponse(BaseModel):
    """OpenAI-compatible error response model."""

    error: dict = Field(..., description='Error details')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    try:
        config.validate()
        setup_civitai()
        print(f'✅ CivitAI to OpenAI Proxy started successfully')
        print(f'📡 Server will run on {config.HOST}:{config.PORT}')
        print(f'🎨 Using default model: {config.CIVITAI_DEFAULT_MODEL}')
    except ValueError as e:
        print(f'❌ Configuration error: {e}')
        raise
    except Exception as e:
        print(f'❌ Startup error: {e}')
        raise

    yield

    # Shutdown
    print('🛑 CivitAI to OpenAI Proxy shutting down')


app = FastAPI(
    title='CivitAI to OpenAI Proxy',
    description='A proxy server that translates between OpenAI Images API and CivitAI API',
    version='1.0.0',
    lifespan=lifespan,
)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            'error': {'message': 'Request validation failed', 'type': 'invalid_request_error', 'details': exc.errors()}
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions in OpenAI-compatible format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': {
                'message': exc.detail,
                'type': 'api_error' if exc.status_code >= 500 else 'invalid_request_error',
                'code': None,
            }
        },
    )


def translate_openai_to_civitai(request: OpenAIImageRequest) -> dict:
    """
    Translate OpenAI image request format to CivitAI format.

    Args:
        request: OpenAI-formatted image generation request

    Returns:
        Dictionary formatted for CivitAI API
    """
    # Parse size to width and height
    width, height = map(int, request.size.split('x'))

    # Map quality to CivitAI parameters
    # Higher quality = more steps and lower CFG scale for better results
    if request.quality == 'hd':
        cfg_scale = 4.0
        steps = 30
    else:  # "standard"
        cfg_scale = 7.0
        steps = 20

    # Use the configured default CivitAI model
    default_model = config.CIVITAI_DEFAULT_MODEL

    # Adjust negative prompt based on style
    base_negative = 'easynegative, badhandv4, (worst quality, low quality, normal quality), bad-artist, blurry'
    if request.style == 'natural':
        negative_prompt = f'{base_negative}, oversaturated, over-processed, artificial'
    else:  # "vivid"
        negative_prompt = f'{base_negative}, dull, muted colors, flat lighting'

    civitai_input = {
        'model': default_model,
        'params': {
            'prompt': request.prompt,
            'negativePrompt': negative_prompt,
            'scheduler': 'EulerA',
            'steps': steps,
            'cfgScale': cfg_scale,
            'width': width,
            'height': height,
            'seed': -1,  # Random seed
            'clipSkip': 2,
        },
        'additionalNetworks': {},
    }

    return civitai_input


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
    print(f'🔄 Starting to poll job {job_id} (max {config.MAX_POLL_ATTEMPTS} attempts)')
    
    for attempt in range(config.MAX_POLL_ATTEMPTS):
        try:
            response = await civitai.jobs.get(token=token, job_id=job_id)
            
            if 'jobs' not in response or len(response['jobs']) == 0:
                print(f'⚠️ No jobs found in response on attempt {attempt + 1}')
                await asyncio.sleep(config.POLL_INTERVAL)
                continue
                
            job = response['jobs'][0]
            job_status = job.get('status', 'unknown')
            
            print(f'📊 Attempt {attempt + 1}/{config.MAX_POLL_ATTEMPTS}: Job status = {job_status}')
            
            # Check if job failed
            if job_status in ['Failed', 'Cancelled']:
                error_msg = job.get('error', 'Job failed without specific error message')
                print(f'❌ Job failed with status {job_status}: {error_msg}')
                raise HTTPException(
                    status_code=500, 
                    detail=f'CivitAI job failed: {error_msg}'
                )
            
            # Check if job is complete
            if job.get('result') and len(job['result']) > 0:
                result = job['result'][0]
                if result.get('available'):
                    print(f'✅ Job completed successfully after {attempt + 1} attempts')
                    return response
                else:
                    print(f'⏳ Job has result but image not yet available (attempt {attempt + 1})')
            else:
                print(f'⏳ Job still processing (attempt {attempt + 1})')
            
            # Wait before next poll
            await asyncio.sleep(config.POLL_INTERVAL)
            
        except HTTPException:
            # Re-raise HTTP exceptions (like job failures)
            raise
        except Exception as e:
            print(f'⚠️ Error polling job {job_id} on attempt {attempt + 1}: {e}')
            await asyncio.sleep(config.POLL_INTERVAL)
    
    total_time = config.MAX_POLL_ATTEMPTS * config.POLL_INTERVAL
    print(f'⏰ Job {job_id} timed out after {total_time} seconds')
    raise HTTPException(
        status_code=408, 
        detail=f'Job timed out after {total_time} seconds. Try again or use a simpler prompt.'
    )


def translate_civitai_to_openai(civitai_response: dict, request: OpenAIImageRequest) -> OpenAIImageResponse:
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
        job = civitai_response['jobs'][0]

        if not job.get('result') or len(job['result']) == 0:
            raise ValueError('No results found in CivitAI response')

        result = job['result'][0]

        if not result.get('available'):
            raise ValueError('Generated image is not available')

        # Get the image URL
        image_url = result.get('blobUrl')
        if not image_url:
            raise ValueError('No image URL found in result')

        # Create image data based on requested format
        image_data = OpenAIImageData(
            url=image_url if request.response_format == 'url' else None,
            b64_json=None,  # CivitAI typically returns URLs, not base64
            revised_prompt=request.prompt,  # CivitAI doesn't modify prompts like DALL-E
        )

        # Use job creation time if available, otherwise current time
        created_time = job.get('createdAt', int(time.time()))
        if isinstance(created_time, str):
            # If timestamp is a string, try to parse it
            import datetime

            try:
                dt = datetime.datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                created_time = int(dt.timestamp())
            except:
                created_time = int(time.time())

        return OpenAIImageResponse(created=created_time, data=[image_data])

    except KeyError as e:
        raise ValueError(f'Missing required field in CivitAI response: {e}')
    except Exception as e:
        raise ValueError(f'Failed to parse CivitAI response: {e}')


@app.post('/v1/images/generations', response_model=OpenAIImageResponse)
async def create_image(
    request: OpenAIImageRequest,
    authorization: Optional[str] = Header(None, description='Bearer token for authentication'),
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
        if not config.CIVITAI_API_TOKEN:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='CivitAI API token not configured'
            )

        # Currently only support generating 1 image at a time due to CivitAI limitations
        if request.n > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Multiple image generation not supported. Please set n=1',
            )

        # Validate prompt length
        if len(request.prompt.strip()) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Prompt cannot be empty')

                # Translate request format
        civitai_input = translate_openai_to_civitai(request)
        print(f'🎨 Submitting image generation request: {request.size}, quality={request.quality}, style={request.style}')
        
        # Submit job to CivitAI
        civitai_response = await civitai.image.create(civitai_input, wait=False)
        
        # Extract job details
        job_token = civitai_response['token']
        job_id = civitai_response['jobs'][0]['jobId']
        print(f'📝 Job submitted successfully: ID={job_id}, Token={job_token[:8]}...')

        # Poll for completion
        final_response = await poll_civitai_job(job_token, job_id)

        # Translate response format
        try:
            openai_response = translate_civitai_to_openai(final_response, request)
        except ValueError as e:
            print(f'Response translation error: {e}')
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=f'Failed to process CivitAI response: {str(e)}'
            )

        return openai_response

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f'Request validation error: {str(e)}'
        )
    except civitai.exceptions.CivitAIException as e:
        print(f'CivitAI API error: {e}')
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f'CivitAI API error: {str(e)}')
    except Exception as e:
        print(f'Unexpected error generating image: {e}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='An unexpected error occurred while generating the image',
        )


@app.get('/health')
async def health_check() -> dict:
    """Health check endpoint."""
    return {'status': 'healthy', 'service': 'civitai-openai-proxy'}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
