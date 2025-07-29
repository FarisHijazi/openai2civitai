"""
OpenAI to CivitAI Image API Proxy

A proxy server that translates between OpenAI Images API and CivitAI API.
"""

__version__ = '1.0.0'
__author__ = 'FarisHijazi'

from .server import (
    app,
    OpenAIImageRequest,
    OpenAIChatRequest,
    create_image,
    create_chat_completion,
    health_check,
    poll_civitai_job,
    civitai_image_create,
    civitai_jobs_get,
    translate_openai_to_civitai,
    translate_civitai_to_openai,
    fetch_image_as_base64,
    main
)
from openai.types import ImagesResponse

__all__ = [
    'app',
    'OpenAIImageRequest',
    'OpenAIChatRequest',
    'ImagesResponse',
    'create_image',
    'create_chat_completion',
    'health_check',
    'poll_civitai_job',
    'civitai_image_create',
    'civitai_jobs_get',
    'translate_openai_to_civitai',
    'translate_civitai_to_openai',
    'fetch_image_as_base64',
    'main'
]

