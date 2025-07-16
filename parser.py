import copy
import json
import os
import re
from typing import Any, Dict

import joblib
import requests

memory = joblib.Memory(location="./cache", verbose=0)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "urn:air:sdxl:checkpoint:civitai:101055@128078") # official SDXL
DEFAULT_NEGATIVEPROMPT = os.getenv("DEFAULT_NEGATIVEPROMPT", "easynegative, badhandv4, (worst quality, low quality, normal quality), bad-artist, blurry, ugly, ((bad anatomy)),((bad hands)),((bad proportions)),((duplicate limbs)),((fused limbs)),((interlocking fingers)),((poorly drawn face)), signature, watermark, artist logo, patreon logo")
INPUT_TEMPLATE = os.getenv("INPUT_TEMPLATE", "{prompt}")


GENERATION_PARAMS_DEFAULTS = {
    "negativePrompt": DEFAULT_NEGATIVEPROMPT,
    "scheduler": "EulerA",
    "steps": 25,
    "cfgScale": 4,
    "width": 1024,
    "height": 1024,
    "clipSkip": 2,
}


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
    user_input = INPUT_TEMPLATE.format(prompt=user_input)
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
    model = generation_data["other_metadata"].get("model", DEFAULT_MODEL)
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
    params["prompt"] = generation_data["prompt"]
    params["negativePrompt"] = generation_data["negativePrompt"]

    generation_input = dict(
        model=generation_data["other_metadata"].get("model", DEFAULT_MODEL),
        params=params,
        additionalNetworks=generation_data["additionalNetworks"],
    )
    return generation_input
