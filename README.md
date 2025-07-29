# CivitAI to OpenAI Proxy

A simple proxy server that translates [OpenAI DALL-E API](https://platform.openai.com/docs/api-reference/images) requests to [CivitAI's image generation API](https://developer.civitai.com/docs/api/introduction).

This guide assumes you are familiar with both APIs. It only covers how to run and use this proxy.

**Why does this exist?**: most apps that use image generation use OpenAI's API, but CivitAI 
has many better models. This proxy allows existing code-bases that use OpenAI image 
generation to simply change the API endpoint to use this proxy. For example Open-WebUI does supports OpenAI image generation but not CivitAI.

To adapt your existing OpenAI-based application, you only need to change the client initialization:

```diff
-client = OpenAI()
+client = OpenAI(
+    api_key="not-needed", # The proxy uses CIVITAI_API_TOKEN from your environment
+    base_url="http://localhost:8000/v1"
+)
```

# Setup and Usage

## 1. Configure Environment
Create a `.env` file or export the following environment variables:

```bash
export CIVITAI_API_TOKEN="your_api_token_here"

# Optional: Override default settings
export CIVITAI_DEFAULT_MODEL="urn:air:sdxl:checkpoint:civitai:257749@290640"
export HOST="0.0.0.0"
export PORT="8000"
```

## 2. Run with `pip`

```
pip install git+https://github.com/FarisHijazi/openai2civitai.git
```

## 3. Test with cURL
Open a new terminal and send a test request:

```bash
# replicating: https://civitai.com/images/81872053
curl -X POST "http://localhost:8000/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "masterpiece, cinematic, ultra-detailed, realistic digital art, shattered planet divided clearly into twelve distinct biome fragments, each fragment representing a unique biome: dense rainforest, icy tundra, arid desert, lush grasslands, volcanic wasteland, deep ocean, floating islands, mystical forest, rocky mountains, luminous crystal caves, urban futuristic cityscape, ethereal astral realm, visually cohesive symbolic formation, vibrant colors, intricate details, glowing edges, symbolic design, cosmic background, trending on artstation, high contrast, high resolution 4K, cinematic lighting\nSteps: 30, CFG scale: 7.5, Sampler: Undefined, Seed: 1878377707, workflow: txt2img, extra: [object Object], Size: 832x1216, draft: false, width: 832, height: 1216, fluxMode: urn:air:flux1:checkpoint:civitai:618692@691639, quantity: 4, baseModel: Flux1, Created Date: 2025-06-12T1037:36.6295151Z, Clip skip: 2\nAdditional networks: urn:air:flux1:lora:civitai:1384710@1564733*0.1",
    "size": "1024x1024"
  }'
```

## Configuration

The proxy is configured via environment variables:

| Variable              | Default                                                 | Description                                  |
|-----------------------|---------------------------------------------------------|----------------------------------------------|
| `CIVITAI_API_TOKEN`   | **(required)**                                          | Your CivitAI API token.                      |
| `CIVITAI_DEFAULT_MODEL`| `urn:air:sdxl:checkpoint:civitai:257749@290640`         | The default CivitAI model to use.            |
| `HOST`                | `0.0.0.0`                                               | Server host.                                 |
| `PORT`                | `8000`                                                  | Server port.                                 |
| `MAX_POLL_ATTEMPTS`   | `200`                                                   | How many times to check if an image is ready. |
| `POLL_INTERVAL`       | `2`                                                     | Seconds to wait between checks.              |

## Limitations

- Supports only image generation (`/v1/images/generations`).
- The `quality` and `style` parameters are ignored.

## Health Check

To verify the server is running, use the `/health` endpoint:

```bash
curl http://localhost:8000/health
```

## Testing

A test script is included to verify the proxy's functionality:

```bash
python test_proxy.py
```

## Files

- `app.py` - The main proxy server application.
- `config.py` - Handles configuration from environment variables.
- `test_proxy.py` - Script to test the proxy.
- `debug_civitai.py` - Script for testing the direct CivitAI API connection.
- `example_civitai_api_call.py` - An example of a direct call to the CivitAI API.
- `requirements.txt` - Required Python packages.

## TODOs

- [ ] create test suite, for openai SDK, and for jailbreak
