"""generate_images.py

Cloudflare Workers AI Image Generation Pipeline
Model: @cf/black-forest-labs/flux-2-klein-4b

Six-Batch Production Architecture:
- Batch 1: 48 photorealistic images, index 01 (Account 1)
- Batch 2: 48 photorealistic images, index 02 (Account 2)
- Batch 3: 48 photorealistic images, index 03 (Account 3)
- Batch 4: 48 hand-drawn sketch images, index 01 (Account 4)
- Batch 5: 48 hand-drawn sketch images, index 02 (Account 5)
- Batch 6: 48 hand-drawn sketch images, index 03 (Account 6)
Total: 288 images (48 categories × 3 photorealistic + 48 categories × 3 sketch)

Features:
- Dedicated batch-to-account mapping via credentials.txt (sole credential source)
- Independent stateless API calls (1:1, 1024x1024, zero conversational history)
- Direct original JPEG byte preservation (zero recompression or transcoding)
- Explicit detection and rejection of unexpected non-JPEG payloads
- Direct atomic writes to .jpg with no redundant raw_outputs/ duplication
- Manifest checkpointing after every single image
- Safe interruption & resume (never overwrites valid completed files)
- Strict safety-filter compliance (Code 3030 blocked, no retries)
- Bounded retries for transient infrastructure errors
- CLI support: --batch <1-6|all>, --dry-run, --status, --mock, --pilot
"""

import os
import sys
import re
import json
import time
import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import requests
from PIL import Image

# Model specifications
MODEL_ID = "@cf/black-forest-labs/flux-2-klein-4b"
BASE_API_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}"

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = WORKSPACE_DIR / "prompts"
IMAGES_DIR = WORKSPACE_DIR / "images"
ARCHIVE_DIR = WORKSPACE_DIR / "archive"
PILOT_RESULTS_DIR = ARCHIVE_DIR / "pilot_results"
DEFAULT_CATEGORIES_FILE = WORKSPACE_DIR / "categories.json"
DEFAULT_CREDENTIALS_FILE = WORKSPACE_DIR / "credentials.txt"
DEFAULT_MANIFEST_FILE = WORKSPACE_DIR / "dataset_manifest.json"

# Quota & Retry defaults
NEURONS_PER_IMAGE_ESTIMATE = 105
MAX_TRANSIENT_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0

# ---------------------------------------------------------------------------
# Six-Batch Definitions
# ---------------------------------------------------------------------------

BATCH_DEFINITIONS: Dict[int, Dict[str, Any]] = {
    1: {
        "batch_number": 1,
        "style": "photorealistic",
        "index": 1,
        "account_label": "ACCOUNT_1",
        "description": "48 photorealistic images, index 01",
    },
    2: {
        "batch_number": 2,
        "style": "photorealistic",
        "index": 2,
        "account_label": "ACCOUNT_2",
        "description": "48 photorealistic images, index 02",
    },
    3: {
        "batch_number": 3,
        "style": "photorealistic",
        "index": 3,
        "account_label": "ACCOUNT_3",
        "description": "48 photorealistic images, index 03",
    },
    4: {
        "batch_number": 4,
        "style": "sketch",
        "index": 1,
        "account_label": "ACCOUNT_4",
        "description": "48 sketch images, index 01",
    },
    5: {
        "batch_number": 5,
        "style": "sketch",
        "index": 2,
        "account_label": "ACCOUNT_5",
        "description": "48 sketch images, index 02",
    },
    6: {
        "batch_number": 6,
        "style": "sketch",
        "index": 3,
        "account_label": "ACCOUNT_6",
        "description": "48 sketch images, index 03",
    },
}


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    """Base exception for the image generation pipeline."""
    pass


class AuthError(PipelineError):
    """Fatal authentication or authorization failure (HTTP 401/403)."""
    pass


class SafetyBlockedError(PipelineError):
    """Platform content moderation / safety filter block (HTTP 400 Code 3030)."""
    def __init__(self, message: str, code: int = 3030):
        super().__init__(message)
        self.code = code


class QuotaExceededError(PipelineError):
    """Daily neuron quota or platform rate limit exhausted (HTTP 429)."""
    pass


class TransientAPIError(PipelineError):
    """Temporary service failure eligible for bounded retry (5xx, timeout, network)."""
    pass


class ImageValidationError(PipelineError):
    """Decoded or saved image failed technical validation checks."""
    pass


# ---------------------------------------------------------------------------
# Security & Sanitization
# ---------------------------------------------------------------------------

def sanitize_message(msg: str, account_id: Optional[str] = None, api_token: Optional[str] = None) -> str:
    """Removes sensitive credentials and identifiers from error and log messages."""
    sanitized = str(msg)
    if api_token:
        sanitized = sanitized.replace(api_token, "[REDACTED_TOKEN]")
    if account_id:
        sanitized = sanitized.replace(account_id, "[REDACTED_ACCOUNT_ID]")
    return sanitized


def parse_credentials_file(path: Path = DEFAULT_CREDENTIALS_FILE) -> Dict[str, Dict[str, str]]:
    """
    Parses sections from credentials.txt without logging or exposing values.
    Returns dict: { 'ACCOUNT_1': {'CLOUDFLARE_ACCOUNT_ID': '...', 'CLOUDFLARE_API_TOKEN': '...'}, ... }
    """
    if not path.exists():
        return {}

    accounts: Dict[str, Dict[str, str]] = {}
    current_label: Optional[str] = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_label = stripped[1:-1].strip()
            accounts[current_label] = {}
        elif "=" in stripped and current_label:
            k, v = stripped.split("=", 1)
            accounts[current_label][k.strip()] = v.strip()

    return accounts


def load_credentials_for_batch(
    batch_number: int,
    credentials_file: Path = DEFAULT_CREDENTIALS_FILE,
) -> Tuple[str, str, str]:
    """
    Loads the designated credentials for a specific batch.
    Maps Batch N to ACCOUNT_N strictly from credentials.txt.
    Returns (account_id, api_token, account_label).
    Never prints or exposes the actual secrets.
    """
    binfo = BATCH_DEFINITIONS.get(batch_number)
    if not binfo:
        raise ValueError(f"Invalid batch number: {batch_number}. Must be 1 to 6.")

    account_label = binfo["account_label"]
    if not credentials_file.exists():
        raise AuthError(f"Credentials file not found at: {credentials_file.name}")

    accounts = parse_credentials_file(credentials_file)

    if account_label in accounts:
        acc_data = accounts[account_label]
        account_id = acc_data.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        api_token = acc_data.get("CLOUDFLARE_API_TOKEN", "").strip()
        if account_id and api_token:
            return account_id, api_token, account_label

    raise AuthError(
        f"Missing or incomplete credentials for {account_label} (Batch {batch_number}) in {credentials_file.name}."
    )


def load_credentials(
    account_label: str = "ACCOUNT_1",
    credentials_file: Path = DEFAULT_CREDENTIALS_FILE,
) -> Tuple[str, str]:
    """
    Loads credentials for a specified account (defaults to ACCOUNT_1) directly from credentials.txt.
    Eliminates all runtime dependencies on .env and python-dotenv.
    Returns (account_id, api_token).
    Never prints or exposes the actual secrets.
    """
    if not credentials_file.exists():
        raise AuthError(f"Credentials file not found at: {credentials_file.name}")

    accounts = parse_credentials_file(credentials_file)
    if account_label not in accounts:
        raise AuthError(f"Section [{account_label}] not found in {credentials_file.name}")

    acc_data = accounts[account_label]
    account_id = acc_data.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    api_token = acc_data.get("CLOUDFLARE_API_TOKEN", "").strip()

    if not account_id:
        raise AuthError(f"CLOUDFLARE_ACCOUNT_ID is not configured or empty for {account_label} in {credentials_file.name}")
    if not api_token:
        raise AuthError(f"CLOUDFLARE_API_TOKEN is not configured or empty for {account_label} in {credentials_file.name}")

    return account_id, api_token


# ---------------------------------------------------------------------------
# Prompt Management
# ---------------------------------------------------------------------------

def get_prompt_path_for_style(style: str) -> Path:
    """Resolves the prompt template file path for a visual style."""
    if style == "photorealistic":
        return PROMPTS_DIR / "photorealistic.txt"
    elif style == "sketch":
        return PROMPTS_DIR / "sketch.txt"
    else:
        raise ValueError(f"Unknown image style: '{style}'. Supported: 'photorealistic', 'sketch'")


def load_prompt_template(style: str) -> Tuple[str, str]:
    """
    Loads the prompt template for the style and calculates its SHA-256 checksum.
    Returns (template_text, sha256_checksum).
    """
    path = get_prompt_path_for_style(style)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found at: {path}")
    raw_bytes = path.read_bytes()
    checksum = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8")
    return text, checksum


def build_prompt_for_object(master_prompt: str, target_object: str) -> str:
    """
    Replaces only the OBJECT declaration in the master prompt.
    Preserves all remaining lines, whitespace, and formatting verbatim.
    """
    pattern = r"^OBJECT:\s*.*$"
    match = re.search(pattern, master_prompt, flags=re.MULTILINE)
    if not match:
        raise ValueError("Could not find 'OBJECT: <value>' declaration in master prompt.")

    replaced_prompt = re.sub(
        pattern,
        f"OBJECT: {target_object}",
        master_prompt,
        count=1,
        flags=re.MULTILINE,
    )
    return replaced_prompt


# ---------------------------------------------------------------------------
# Perceptual & Cryptographic Hashes
# ---------------------------------------------------------------------------

def calculate_sha256(data: bytes) -> str:
    """Calculates SHA-256 hex digest for binary data."""
    return hashlib.sha256(data).hexdigest()


def calculate_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """
    Calculates a 64-bit difference hash (dHash) for perceptual similarity comparison.
    Converts image to grayscale, resizes to (hash_size + 1, hash_size),
    and compares adjacent horizontal pixel values.
    """
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    raw_pixels = list(gray.get_flattened_data()) if hasattr(gray, "get_flattened_data") else list(gray.getdata())

    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            idx = row * (hash_size + 1) + col
            diff.append(raw_pixels[idx] > raw_pixels[idx + 1])

    decimal_val = 0
    for bit in diff:
        decimal_val = (decimal_val << 1) | int(bit)
    return f"{decimal_val:016x}"


# ---------------------------------------------------------------------------
# Image Decoding, Format Validation & Atomic Persistence
# ---------------------------------------------------------------------------

def validate_original_jpeg(
    raw_bytes: bytes,
    expected_width: int = 1024,
    expected_height: int = 1024,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Validates the raw image payload from Cloudflare Workers AI.
    Verifies that the payload is genuine JPEG encoded data matching expected dimensions.
    Returns (raw_bytes, metadata) with ZERO recompression, preserving exact upstream bytes.
    Detects and reports unexpected non-JPEG payloads (e.g., PNG, WebP) with descriptive errors.
    """
    import io

    if not raw_bytes or len(raw_bytes) == 0:
        raise ImageValidationError("Received empty image payload (0 bytes).")

    # 1. Check JPEG magic SOI header marker (FF D8 FF)
    if not raw_bytes.startswith(b"\xff\xd8\xff"):
        if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ImageValidationError("Unexpected non-JPEG format: Received PNG payload instead of expected JPEG.")
        elif raw_bytes.startswith(b"RIFF") and b"WEBP" in raw_bytes[:16]:
            raise ImageValidationError("Unexpected non-JPEG format: Received WebP payload instead of expected JPEG.")
        elif raw_bytes.startswith(b"GIF87a") or raw_bytes.startswith(b"GIF89a"):
            raise ImageValidationError("Unexpected non-JPEG format: Received GIF payload instead of expected JPEG.")
        else:
            raise ImageValidationError(
                f"Payload lacks JPEG SOI header marker (FF D8 FF). First 8 bytes: {raw_bytes[:8].hex()}"
            )

    # 2. Decode and verify with Pillow in read-only mode (zero recompression)
    try:
        with Image.open(io.BytesIO(raw_bytes)) as test_img:
            actual_format = test_img.format
            if actual_format != "JPEG":
                raise ImageValidationError(
                    f"Unexpected non-JPEG format detected by decoder: '{actual_format}' (expected 'JPEG')"
                )
            actual_mode = test_img.mode
            width, height = test_img.size

        # Open and load fully to verify uncorrupted stream and compute perceptual dHash
        with Image.open(io.BytesIO(raw_bytes)) as load_img:
            load_img.load()
            dhash_val = calculate_dhash(load_img)

    except ImageValidationError:
        raise
    except Exception as e:
        raise ImageValidationError(f"Failed to decode or verify JPEG image stream: {e}")

    # 3. Verify dimensions & aspect ratio
    if width != expected_width or height != expected_height:
        raise ImageValidationError(
            f"Unexpected image dimensions: {width}x{height} (expected {expected_width}x{expected_height})"
        )
    if width != height:
        raise ImageValidationError(f"Non-square aspect ratio detected: {width}x{height}")

    # 4. Compute cryptographic metadata on exact original bytes
    jpeg_sha = calculate_sha256(raw_bytes)

    metadata: Dict[str, Any] = {
        "format": "JPEG",
        "mode": actual_mode,
        "byte_count": len(raw_bytes),
        "jpeg_sha256": jpeg_sha,
        "dhash": dhash_val,
        "width": width,
        "height": height,
        "is_square": True,
        "is_1024x1024": True,
    }

    # Return pristine, original raw_bytes directly with zero recompression
    return raw_bytes, metadata


def validate_and_transcode_to_png(
    raw_bytes: bytes,
    expected_width: int = 1024,
    expected_height: int = 1024,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    [Legacy / Pilot Reference Function]
    Transcodes raw bytes to PNG. Preserved for backward compatibility.
    """
    import io

    if not raw_bytes or len(raw_bytes) == 0:
        raise ImageValidationError("Received empty image payload (0 bytes).")

    # 1. Decode original bytes with Pillow
    try:
        with Image.open(io.BytesIO(raw_bytes)) as raw_img:
            raw_format = raw_img.format
            raw_mode = raw_img.mode
            width, height = raw_img.size
            raw_img.load()

            if raw_mode not in ("RGB", "RGBA"):
                clean_img = raw_img.convert("RGB")
            else:
                clean_img = raw_img.copy()
    except Exception as e:
        raise ImageValidationError(f"Failed to decode raw image stream: {e}")

    # 2. Verify dimensions & aspect ratio
    if width != expected_width or height != expected_height:
        raise ImageValidationError(
            f"Unexpected image dimensions: {width}x{height} (expected {expected_width}x{expected_height})"
        )
    if width != height:
        raise ImageValidationError(f"Non-square aspect ratio detected: {width}x{height}")

    # 3. Transcode to genuine PNG
    png_buffer = io.BytesIO()
    clean_img.save(png_buffer, format="PNG", optimize=True)
    png_bytes = png_buffer.getvalue()

    # 4. Verify genuine PNG signature (89 50 4E 47 0D 0A 1A 0A)
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ImageValidationError("Transcoded image lacks genuine PNG signature header.")

    # 5. Compute metadata
    raw_sha = calculate_sha256(raw_bytes)
    png_sha = calculate_sha256(png_bytes)
    dhash_val = calculate_dhash(clean_img)

    metadata: Dict[str, Any] = {
        "raw_format": raw_format,
        "raw_mode": raw_mode,
        "raw_bytes": len(raw_bytes),
        "raw_sha256": raw_sha,
        "target_format": "PNG",
        "target_mode": clean_img.mode,
        "png_bytes": len(png_bytes),
        "png_sha256": png_sha,
        "dhash": dhash_val,
        "width": width,
        "height": height,
        "is_square": True,
        "is_1024x1024": True,
    }

    return png_bytes, metadata


def save_atomic_image(
    image_bytes: bytes,
    target_path: Path,
    raw_bytes: Optional[bytes] = None,
    raw_path: Optional[Path] = None,
) -> None:
    """
    Saves image bytes atomically using a temporary file and atomic rename.
    Guarantees no half-written or corrupted files on disk.
    Optionally saves raw_bytes to raw_path if explicitly provided.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.parent / f".tmp_{target_path.name}"

    try:
        with open(temp_target, "wb") as f:
            f.write(image_bytes)
        temp_target.replace(target_path)

        if raw_bytes and raw_path:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temp_raw = raw_path.parent / f".tmp_{raw_path.name}"
            with open(temp_raw, "wb") as f:
                f.write(raw_bytes)
            temp_raw.replace(raw_path)

    finally:
        if temp_target.exists():
            try:
                temp_target.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# API Request & Error Handling
# ---------------------------------------------------------------------------

def execute_single_api_call(
    account_id: str,
    api_token: str,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    timeout_seconds: int = 120,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Submits a single, independent API request to Cloudflare Workers AI.
    Never passes conversation history or previous generation context.
    """
    url = BASE_API_URL.format(account_id=account_id, model_id=MODEL_ID)
    headers = {
        "Authorization": f"Bearer {api_token}",
    }

    files: Dict[str, Any] = {
        "prompt": (None, prompt),
        "width": (None, str(width)),
        "height": (None, str(height)),
    }
    if seed is not None:
        files["seed"] = (None, str(seed))

    start_time = time.time()
    try:
        response = requests.post(
            url,
            headers=headers,
            files=files,
            timeout=timeout_seconds,
        )
    except requests.exceptions.Timeout as e:
        raise TransientAPIError(f"Request timed out after {timeout_seconds}s: {e}")
    except requests.exceptions.ConnectionError as e:
        raise TransientAPIError(f"Network connection failed: {e}")
    except requests.exceptions.RequestException as e:
        raise TransientAPIError(f"Request exception: {e}")

    elapsed_time = round(time.time() - start_time, 2)
    status_code = response.status_code
    content_type = response.headers.get("Content-Type", "")

    meta_headers = {
        k: v for k, v in response.headers.items()
        if k.lower() in ["cf-ray", "content-type", "date", "retry-after"]
    }

    if status_code in (401, 403):
        raise AuthError(
            f"Authentication/Authorization failed (HTTP {status_code}). Check Cloudflare API token."
        )

    if status_code == 429:
        retry_after = response.headers.get("Retry-After")
        err_text = response.text[:200]
        if "quota" in err_text.lower() or "neuron" in err_text.lower():
            raise QuotaExceededError("Daily neuron quota exhausted (HTTP 429).")
        raise TransientAPIError(
            f"Rate limit encountered (HTTP 429). Retry-After: {retry_after or 'unspecified'}"
        )

    if status_code >= 500:
        raise TransientAPIError(f"Cloudflare server error (HTTP {status_code}): {response.text[:150]}")

    if status_code == 400:
        try:
            err_json = response.json()
            errors = err_json.get("errors", [])
            for err in errors:
                err_code = err.get("code")
                err_msg = str(err.get("message", ""))
                if err_code == 3030 or "flagged" in err_msg.lower() or "nsfw" in err_msg.lower():
                    raise SafetyBlockedError(
                        f"Cloudflare Content Safety Filter Block (Code {err_code}): {err_msg}",
                        code=err_code or 3030,
                    )
            raise PipelineError(f"Bad Request (HTTP 400): {errors}")
        except SafetyBlockedError:
            raise
        except Exception as e:
            if isinstance(e, PipelineError):
                raise
            raise PipelineError(f"Bad Request (HTTP 400): {response.text[:200]}")

    if status_code != 200:
        raise PipelineError(f"Unexpected HTTP status {status_code}: {response.text[:200]}")

    raw_bytes: Optional[bytes] = None
    is_png = response.content.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = response.content.startswith(b"\xff\xd8\xff")

    if "image/" in content_type or is_png or is_jpeg:
        raw_bytes = response.content
        resp_type = "raw_binary"
    else:
        try:
            data = response.json()
        except Exception:
            raise ImageValidationError(
                f"Failed to parse JSON response. Content-Type: {content_type}. Preview: {response.content[:80]}"
            )

        if not data.get("success", False) and "errors" in data:
            errors = data.get("errors", [])
            for err in errors:
                if err.get("code") == 3030 or "flagged" in str(err.get("message", "")).lower():
                    raise SafetyBlockedError(f"Safety filter rejection: {err}")
            raise PipelineError(f"Cloudflare API error: {errors}")

        result = data.get("result")
        if isinstance(result, dict) and "image" in result:
            raw_bytes = base64.b64decode(result["image"])
            resp_type = "json_base64_result_image"
        elif isinstance(result, str):
            raw_bytes = base64.b64decode(result)
            resp_type = "json_base64_result"
        elif "image" in data and isinstance(data["image"], str):
            raw_bytes = base64.b64decode(data["image"])
            resp_type = "json_base64_image"
        else:
            raise ImageValidationError(f"Could not locate image payload in JSON. Keys: {list(data.keys())}")

    request_metadata = {
        "status_code": status_code,
        "elapsed_seconds": elapsed_time,
        "response_type": resp_type,
        "headers": meta_headers,
    }

    return raw_bytes, request_metadata


def generate_single_image_with_resilience(
    account_id: str,
    api_token: str,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    mock: bool = False,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Executes a single generation with bounded exponential backoff retries
    ONLY for transient errors. Strict no-retry on safety blocks (Code 3030),
    auth failures, or quota limits.
    """
    if mock:
        import io
        time.sleep(0.01)
        if "TRIGGER_SAFETY_BLOCK" in prompt:
            raise SafetyBlockedError("Simulated safety filter rejection (Code 3030)")
        if "TRIGGER_AUTH_ERROR" in prompt:
            raise AuthError("Simulated auth failure (HTTP 401)")
        if "TRIGGER_QUOTA_ERROR" in prompt:
            raise QuotaExceededError("Simulated quota exhaustion (HTTP 429)")

        mock_img = Image.new("RGB", (width, height), color=(120, 140, 180))
        buf = io.BytesIO()
        mock_img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()
        return raw_bytes, {
            "status_code": 200,
            "elapsed_seconds": 0.01,
            "response_type": "mock_synthetic_jpeg",
            "headers": {"cf-ray": "mock-ray-12345"},
        }

    retries = 0
    backoff = INITIAL_BACKOFF_SECONDS

    while True:
        try:
            return execute_single_api_call(
                account_id=account_id,
                api_token=api_token,
                prompt=prompt,
                width=width,
                height=height,
                seed=seed,
            )
        except SafetyBlockedError:
            raise
        except (AuthError, QuotaExceededError):
            raise
        except TransientAPIError as e:
            retries += 1
            if retries > MAX_TRANSIENT_RETRIES:
                raise TransientAPIError(
                    f"Exceeded max retries ({MAX_TRANSIENT_RETRIES}) for transient error: {e}"
                )
            print(f"    [TRANSIENT RETRY {retries}/{MAX_TRANSIENT_RETRIES}] Backoff {backoff:.1f}s... Reason: {e}")
            time.sleep(backoff)
            backoff *= 2.0


# ---------------------------------------------------------------------------
# Progress Manifest Tracking (Six-Batch Aware)
# ---------------------------------------------------------------------------

def initialize_empty_manifest() -> Dict[str, Any]:
    """Generates a clean manifest structure for the 6-batch dataset."""
    batch_stats = {}
    for b_num, b_info in BATCH_DEFINITIONS.items():
        batch_stats[f"batch_{b_num}"] = {
            "batch_number": b_num,
            "style": b_info["style"],
            "index": b_info["index"],
            "account_label": b_info["account_label"],
            "total_target": 48,
            "completed": 0,
            "pending": 48,
            "blocked": 0,
            "failed": 0,
        }

    return {
        "version": "2.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "target_summary": {
            "total_target": 288,
            "batches_count": 6,
            "categories_count": 48,
        },
        "batch_statistics": batch_stats,
        "overall_statistics": {
            "photorealistic_completed": 0,
            "sketch_completed": 0,
            "total_completed": 0,
            "blocked_safety": 0,
            "failed": 0,
        },
        "records": {},
    }


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_FILE) -> Dict[str, Any]:
    """Loads existing progress manifest or initializes a clean six-batch manifest."""
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "batch_statistics" in data:
                    return data
        except Exception:
            pass

    return initialize_empty_manifest()


def update_manifest_statistics(manifest: Dict[str, Any], categories_count: int = 48) -> None:
    """Recomputes manifest summary statistics across all six batches and overall."""
    batch_stats = {}
    for b_num, b_info in BATCH_DEFINITIONS.items():
        batch_stats[f"batch_{b_num}"] = {
            "batch_number": b_num,
            "style": b_info["style"],
            "index": b_info["index"],
            "account_label": b_info["account_label"],
            "total_target": categories_count,
            "completed": 0,
            "pending": categories_count,
            "blocked": 0,
            "failed": 0,
        }

    photo_done = 0
    sketch_done = 0
    total_blocked = 0
    total_failed = 0

    for rec in manifest["records"].values():
        b_num = rec.get("batch")
        b_key = f"batch_{b_num}" if b_num else None
        status = rec.get("status")
        style = rec.get("style")

        if b_key and b_key in batch_stats:
            if status == "COMPLETED":
                batch_stats[b_key]["completed"] += 1
                batch_stats[b_key]["pending"] = max(0, batch_stats[b_key]["pending"] - 1)
            elif status == "BLOCKED_SAFETY_FILTER":
                batch_stats[b_key]["blocked"] += 1
                batch_stats[b_key]["pending"] = max(0, batch_stats[b_key]["pending"] - 1)
            elif status == "FAILED":
                batch_stats[b_key]["failed"] += 1
                batch_stats[b_key]["pending"] = max(0, batch_stats[b_key]["pending"] - 1)

        if status == "COMPLETED":
            if style == "photorealistic":
                photo_done += 1
            elif style == "sketch":
                sketch_done += 1
        elif status == "BLOCKED_SAFETY_FILTER":
            total_blocked += 1
        elif status == "FAILED":
            total_failed += 1

    manifest["batch_statistics"] = batch_stats
    manifest["overall_statistics"] = {
        "photorealistic_completed": photo_done,
        "sketch_completed": sketch_done,
        "total_completed": photo_done + sketch_done,
        "blocked_safety": total_blocked,
        "failed": total_failed,
    }


def save_manifest(
    manifest: Dict[str, Any],
    manifest_path: Path = DEFAULT_MANIFEST_FILE,
    categories_count: int = 48,
) -> None:
    """Atomically saves manifest data to disk."""
    update_manifest_statistics(manifest, categories_count=categories_count)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.parent / f".tmp_{manifest_path.name}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    temp_path.replace(manifest_path)


# ---------------------------------------------------------------------------
# Task Generation & Batch Execution
# ---------------------------------------------------------------------------

def get_batch_tasks(batch_number: int, categories: List[str]) -> List[Dict[str, Any]]:
    """Generates the list of 48 planned tasks for a given batch number."""
    binfo = BATCH_DEFINITIONS[batch_number]
    tasks = []
    for cat in categories:
        target_filename = f"{cat}_{binfo['index']:02d}.jpg"
        target_path = IMAGES_DIR / binfo["style"] / cat / target_filename
        tasks.append({
            "batch": batch_number,
            "style": binfo["style"],
            "category": cat,
            "index": binfo["index"],
            "account_label": binfo["account_label"],
            "target_path": target_path,
            "key": f"{binfo['style']}:{cat}:{binfo['index']:02d}",
        })
    return tasks


def execute_batch(
    batch_number: int,
    categories: List[str],
    manifest_path: Path = DEFAULT_MANIFEST_FILE,
    credentials_file: Path = DEFAULT_CREDENTIALS_FILE,
    mock: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Executes or simulates a single logical batch (48 images).
    Enforces independent requests, original JPEG byte preservation, atomic saves,
    manifest checkpointing after every image, and non-overwrite of existing files.
    """
    binfo = BATCH_DEFINITIONS[batch_number]
    tasks = get_batch_tasks(batch_number, categories)
    manifest = load_manifest(manifest_path)

    print("=" * 70)
    print(f"BATCH {batch_number}/6 EXECUTION")
    print(f"Style:       {binfo['style'].upper()}")
    print(f"Image Index: {binfo['index']:02d}")
    print(f"Account:     {binfo['account_label']}")
    print(f"Task Count:  {len(tasks)} images")
    print(f"Mode:        {'DRY RUN' if dry_run else ('MOCK TEST' if mock else 'LIVE GENERATION')}")
    print("=" * 70)

    if dry_run:
        print(f"\n[DRY RUN SUMMARY for Batch {batch_number}]")
        print(f"  Visual Style:       {binfo['style']}")
        print(f"  Image Index:        {binfo['index']:02d}")
        print(f"  Assigned Account:   {binfo['account_label']} (loaded from {credentials_file.name})")
        print(f"  Planned Targets:    {len(tasks)} categories")
        print(f"  Destination Pattern: images/{binfo['style']}/<category>/<category>_{binfo['index']:02d}.jpg")
        print("\n  Sample Planned Tasks (first 3):")
        for t in tasks[:3]:
            print(f"    - [{t['key']}] -> {t['target_path']}")
        print(f"    ... ({len(tasks) - 3} more tasks)\n")
        print("  -> Dry run complete. 0 API calls made. No files modified.")
        return manifest

    # Load credentials for this specific batch
    account_id, api_token, active_label = ("", "", binfo["account_label"])
    if not mock:
        account_id, api_token, active_label = load_credentials_for_batch(
            batch_number, credentials_file=credentials_file
        )
        print(f"  -> Authenticated using {active_label} for Batch {batch_number}.")

    template_text, prompt_sha = load_prompt_template(binfo["style"])

    for idx, task in enumerate(tasks, start=1):
        target_path: Path = task["target_path"]
        key: str = task["key"]
        cat: str = task["category"]

        # 1. Existing-file protection & resume check
        if target_path.exists() and target_path.stat().st_size > 0:
            try:
                raw_check = target_path.read_bytes()
                if raw_check.startswith(b"\xff\xd8\xff"):
                    with Image.open(target_path) as chk_img:
                        if chk_img.format == "JPEG" and chk_img.size == (1024, 1024):
                            print(f"  [{idx:02d}/{len(tasks)}] Skipping already completed valid image: {target_path.name}")
                            if key not in manifest["records"] or manifest["records"][key].get("status") != "COMPLETED":
                                manifest["records"][key] = {
                                    "batch": batch_number,
                                    "style": binfo["style"],
                                    "category": cat,
                                    "index": binfo["index"],
                                    "account_label": active_label,
                                    "status": "COMPLETED",
                                    "filepath": str(target_path),
                                    "jpeg_sha256": calculate_sha256(raw_check),
                                    "dhash": calculate_dhash(chk_img),
                                    "width": 1024,
                                    "height": 1024,
                                    "format": "JPEG",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                                save_manifest(manifest, manifest_path, categories_count=len(categories))
                            continue
            except Exception:
                pass

        # 2. Build unique prompt
        prompt = build_prompt_for_object(template_text, cat)
        print(f"  [{idx:02d}/{len(tasks)}] Generating {key} using {active_label}...")

        record: Dict[str, Any] = {
            "batch": batch_number,
            "style": binfo["style"],
            "category": cat,
            "index": binfo["index"],
            "account_label": active_label,
            "prompt_sha256": prompt_sha,
            "target_filepath": str(target_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            raw_bytes, req_meta = generate_single_image_with_resilience(
                account_id=account_id,
                api_token=api_token,
                prompt=prompt,
                mock=mock,
            )

            # Validate original JPEG bytes directly with ZERO recompression
            orig_bytes, val_meta = validate_original_jpeg(raw_bytes)

            # Save exact original bytes directly as genuine .jpg
            save_atomic_image(
                image_bytes=orig_bytes,
                target_path=target_path,
            )

            record["status"] = "COMPLETED"
            record["latency_seconds"] = req_meta["elapsed_seconds"]
            record["validation"] = val_meta
            record["filepath"] = str(target_path)
            record["jpeg_sha256"] = val_meta["jpeg_sha256"]
            record["dhash"] = val_meta["dhash"]
            record["format"] = "JPEG"
            print(f"     [OK] Saved original JPEG ({val_meta['byte_count']:,} B) in {req_meta['elapsed_seconds']}s")

        except SafetyBlockedError as sbe:
            clean_msg = sanitize_message(str(sbe), account_id, api_token)
            print(f"     [BLOCKED - SAFETY FILTER] Code {sbe.code}: {clean_msg}")
            record["status"] = "BLOCKED_SAFETY_FILTER"
            record["error"] = clean_msg
            record["error_code"] = sbe.code
            record["manual_review_required"] = True

        except QuotaExceededError as qe:
            clean_msg = sanitize_message(str(qe), account_id, api_token)
            print(f"     [QUOTA EXHAUSTED for {active_label}] {clean_msg}")
            record["status"] = "FAILED"
            record["error"] = clean_msg
            manifest["records"][key] = record
            save_manifest(manifest, manifest_path, categories_count=len(categories))
            raise

        except Exception as ex:
            clean_msg = sanitize_message(str(ex), account_id, api_token)
            print(f"     [FAILED] {clean_msg}")
            record["status"] = "FAILED"
            record["error"] = clean_msg

        # Checkpoint manifest after EVERY single image
        manifest["records"][key] = record
        save_manifest(manifest, manifest_path, categories_count=len(categories))

    print(f"\nBatch {batch_number} execution pass completed.")
    save_manifest(manifest, manifest_path, categories_count=len(categories))
    return manifest


def execute_all_batches(
    categories: List[str],
    manifest_path: Path = DEFAULT_MANIFEST_FILE,
    credentials_file: Path = DEFAULT_CREDENTIALS_FILE,
    mock: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Executes all six batches sequentially (Batch 1 through Batch 6).
    Enforces that each batch operates under its designated account credential.
    """
    print("=" * 70)
    print("EXECUTING ALL SIX PRODUCTION BATCHES (288 IMAGES)")
    print("Batch 1 -> Batch 2 -> Batch 3 -> Batch 4 -> Batch 5 -> Batch 6")
    print("=" * 70)

    manifest = load_manifest(manifest_path)
    for b_num in range(1, 7):
        manifest = execute_batch(
            batch_number=b_num,
            categories=categories,
            manifest_path=manifest_path,
            credentials_file=credentials_file,
            mock=mock,
            dry_run=dry_run,
        )

    return manifest


# ---------------------------------------------------------------------------
# Status Inspection
# ---------------------------------------------------------------------------

def display_status(
    manifest_path: Path = DEFAULT_MANIFEST_FILE,
    categories_count: int = 48,
) -> None:
    """Displays comprehensive six-batch progress status."""
    manifest = load_manifest(manifest_path)
    update_manifest_statistics(manifest, categories_count=categories_count)
    bstats = manifest.get("batch_statistics", {})
    ostats = manifest.get("overall_statistics", {})

    print("=" * 72)
    print("CLOUDFLARE WORKERS AI — SIX-BATCH DATASET STATUS")
    print("=" * 72)

    for b_num in range(1, 7):
        b_key = f"batch_{b_num}"
        s = bstats.get(b_key, {})
        style_label = s.get("style", "").capitalize()
        idx = s.get("index", b_num)
        acc = s.get("account_label", f"ACCOUNT_{b_num}")
        done = s.get("completed", 0)
        tot = s.get("total_target", categories_count)
        pend = s.get("pending", tot - done)
        blk = s.get("blocked", 0)
        fail = s.get("failed", 0)

        flag_str = ""
        if blk > 0:
            flag_str += f" | {blk} BLOCKED (manual review)"
        if fail > 0:
            flag_str += f" | {fail} FAILED"

        print(
            f"Batch {b_num} ({style_label:14} Index {idx:02d} | {acc:9}): "
            f"{done:2d}/{tot:2d} completed | {pend:2d} pending{flag_str}"
        )

    print("-" * 72)
    photo_done = ostats.get("photorealistic_completed", 0)
    sketch_done = ostats.get("sketch_completed", 0)
    total_done = ostats.get("total_completed", 0)
    total_target = 288
    pct = (total_done / total_target * 100) if total_target > 0 else 0

    print(f"Photorealistic Progress: {photo_done:3d} / 144 completed")
    print(f"Sketch Progress:         {sketch_done:3d} / 144 completed")
    print(f"OVERALL DATASET PROGRESS: {total_done:3d} / 288 completed ({pct:5.1f}%)")
    print(f"Blocked Safety Filter:   {ostats.get('blocked_safety', 0)}")
    print(f"Failed Generations:      {ostats.get('failed', 0)}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Pilot Execution Functions
# ---------------------------------------------------------------------------

def run_sketch_pilot(
    output_dir: Path = PILOT_RESULTS_DIR / "sketch",
    targets: Optional[List[str]] = None,
    mock: bool = False,
) -> Dict[str, Any]:
    """Executes the three-image sketch pilot test for dog, airplane, and chair."""
    if targets is None:
        targets = ["dog", "airplane", "chair"]

    account_id, api_token = ("", "")
    if not mock:
        account_id, api_token = load_credentials()

    output_dir.mkdir(parents=True, exist_ok=True)

    template_text, prompt_sha = load_prompt_template("sketch")

    pilot_summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "style": "sketch",
        "prompt_template_sha256": prompt_sha,
        "targets": targets,
        "results": {},
        "all_successful": False,
    }

    all_passed = True
    for idx, target in enumerate(targets, start=1):
        target_jpg = output_dir / f"{target}_01.jpg"

        custom_prompt = build_prompt_for_object(template_text, target)
        try:
            raw_bytes, req_meta = generate_single_image_with_resilience(
                account_id=account_id,
                api_token=api_token,
                prompt=custom_prompt,
                mock=mock,
            )
            orig_bytes, val_meta = validate_original_jpeg(raw_bytes)
            save_atomic_image(
                image_bytes=orig_bytes,
                target_path=target_jpg,
            )
            pilot_summary["results"][target] = {
                "success": True,
                "target_jpg": str(target_jpg),
                "validation": val_meta,
            }
        except Exception as e:
            all_passed = False
            pilot_summary["results"][target] = {"success": False, "error": str(e)}

    pilot_summary["all_successful"] = all_passed
    return pilot_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cloudflare Workers AI Six-Batch Dataset Generator")
    parser.add_argument(
        "--batch",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        default=None,
        help="Specify batch to execute (1 to 6) or 'all' for all six batches sequentially",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display planned tasks and batch mappings without making API calls or modifying files",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display current progress across all six batches and overall dataset",
    )
    parser.add_argument(
        "--categories-file",
        type=Path,
        default=DEFAULT_CATEGORIES_FILE,
        help="JSON file containing the list of categories",
    )
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=DEFAULT_CREDENTIALS_FILE,
        help="Credentials file containing [ACCOUNT_1] through [ACCOUNT_6]",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=DEFAULT_MANIFEST_FILE,
        help="Progress manifest JSON file",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline mock test without live API calls",
    )
    parser.add_argument(
        "--pilot",
        choices=["sketch", "photorealistic"],
        default=None,
        help="Run pilot test for specified style (3 images)",
    )

    args = parser.parse_args()

    # Load categories
    if not args.categories_file.exists():
        print(f"Categories file not found: {args.categories_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.categories_file, "r", encoding="utf-8") as f:
        cat_data = json.load(f)
        categories = cat_data.get("categories", [])

    # Status display
    if args.status:
        display_status(
            manifest_path=args.manifest_file,
            categories_count=len(categories),
        )
        return

    # Pilot execution
    if args.pilot == "sketch":
        summary = run_sketch_pilot(mock=args.mock)
        if not summary.get("all_successful"):
            sys.exit(1)
        return

    # Batch execution
    if args.batch is not None:
        if args.batch == "all":
            execute_all_batches(
                categories=categories,
                manifest_path=args.manifest_file,
                credentials_file=args.credentials_file,
                mock=args.mock,
                dry_run=args.dry_run,
            )
        else:
            batch_num = int(args.batch)
            execute_batch(
                batch_number=batch_num,
                categories=categories,
                manifest_path=args.manifest_file,
                credentials_file=args.credentials_file,
                mock=args.mock,
                dry_run=args.dry_run,
            )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
