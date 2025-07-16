#!/usr/bin/env python3
"""
Test script for the CivitAI to OpenAI proxy server.

This script demonstrates how to use the proxy server to generate images
using OpenAI-compatible requests.
"""

import json
import time
from typing import Optional

import requests


def test_proxy_server(
    base_url: str = "http://localhost:8000",
    prompt: str = "A beautiful landscape with mountains and a lake",
) -> Optional[str]:
    """
    Test the proxy server by generating an image.

    Args:
        base_url: Base URL of the proxy server
        prompt: Text prompt for image generation

    Returns:
        URL of the generated image, or None if failed
    """
    endpoint = f"{base_url}/v1/images/generations"

    # Request payload in OpenAI format
    payload = {
        "prompt": prompt,
        "size": "1024x1024",
        "quality": "hd",
        "style": "vivid",
        "n": 1,
    }

    print(f"🎨 Generating image with prompt: '{prompt}'")
    print(f"📡 Sending request to: {endpoint}")

    try:
        # Send the request
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=360,  # 6 minutes timeout
        )

        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            image_url = result["data"][0]["url"]
            print(f"✅ Success! Image generated: {image_url}")
            return image_url
        else:
            # Print error details
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get(
                    "message", "Unknown error"
                )
                print(f"❌ Error {response.status_code}: {error_message}")
            except json.JSONDecodeError:
                print(f"❌ Error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("⏰ Request timed out. Image generation might take longer than expected.")
        return None
    except requests.exceptions.ConnectionError:
        print("🔌 Could not connect to the proxy server. Make sure it's running.")
        return None
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return None


def check_server_health(base_url: str = "http://localhost:8000") -> bool:
    """
    Check if the proxy server is running and healthy.

    Args:
        base_url: Base URL of the proxy server

    Returns:
        True if server is healthy, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"💚 Server is healthy: {health_data}")
            return True
        else:
            print(f"💔 Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"💔 Could not reach server: {e}")
        return False


def main() -> None:
    """Main test function."""
    print("🚀 Testing CivitAI to OpenAI Proxy Server\n")

    # Check server health first
    if not check_server_health():
        print("\n❌ Server health check failed. Make sure the proxy server is running.")
        print("Start it with: python app.py")
        return

    print("\n" + "=" * 50)

    # Test different prompts
    test_prompts = [
        "A cute cat wearing a wizard hat",
        "A futuristic city with flying cars",
        "A peaceful zen garden with cherry blossoms",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔸 Test {i}/{len(test_prompts)}")
        image_url = test_proxy_server(prompt=prompt)

        # Add a small delay between requests
        if i < len(test_prompts):
            print("⏳ Waiting 3 seconds before next test...")
            time.sleep(3)

    print("\n" + "=" * 50)
    print("🎉 Testing complete!")


if __name__ == "__main__":
    main()
