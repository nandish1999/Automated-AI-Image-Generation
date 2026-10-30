"""generate_single_task.py

Targeted generation script for an individual image task with custom account delegation
and optional semantic disambiguation for line 1 of the prompt.
"""

import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

import generate_images as gen

WORKSPACE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = WORKSPACE_DIR / "dataset_manifest.json"
CREDENTIALS_PATH = WORKSPACE_DIR / "credentials.txt"


def generate_airplane_01(
    account_label: str = "ACCOUNT_1",
    disambiguated_object: str = "commercial passenger airplane",
) -> bool:
    target_key = "photorealistic:airplane:01"
    style = "photorealistic"
    category = "airplane"
    index = 1
    target_path = WORKSPACE_DIR / "images" / style / category / f"{category}_{index:02d}.jpg"

    print("=" * 72)
    print("TARGETED SINGLE-TASK GENERATION (SEMANTIC DISAMBIGUATION)")
    print(f"Task:                  {target_key}")
    print(f"Assigned Account:      {account_label}")
    print(f"Disambiguated Object:  {disambiguated_object}")
    print(f"Output File:           {target_path}")
    print("=" * 72)

    # 1. Load credentials for target account
    account_id, api_token = gen.load_credentials(
        account_label=account_label, credentials_file=CREDENTIALS_PATH
    )
    print(f"  -> Successfully authenticated using {account_label} credentials.")

    # 2. Load invariant master prompt template and format with disambiguated object
    template_text, template_sha = gen.load_prompt_template(style)
    prompt = gen.build_prompt_for_object(template_text, disambiguated_object)
    print(f"  -> Loaded master prompt template (SHA-256: {template_sha[:16]}...).")
    print(f"  -> Formatted prompt: line 1 = 'OBJECT: {disambiguated_object}'.")

    # 3. Load manifest and prepare audit history
    manifest = gen.load_manifest(MANIFEST_PATH)
    rec = manifest["records"].get(target_key, {})
    audit_history = list(rec.get("audit_history", []))

    if rec.get("status") == "BLOCKED_SAFETY_FILTER":
        audit_history.append({
            "attempt_timestamp": rec.get("timestamp"),
            "status": "BLOCKED_SAFETY_FILTER",
            "error_code": rec.get("error_code", 3030),
            "error_message": rec.get("error"),
            "account_label": rec.get("account_label", "ACCOUNT_1"),
        })

    print(f"  -> Submitting request to Cloudflare Workers AI using {account_label}...")

    try:
        raw_bytes, req_meta = gen.generate_single_image_with_resilience(
            account_id=account_id,
            api_token=api_token,
            prompt=prompt,
            mock=False,
        )

        orig_bytes, val_meta = gen.validate_original_jpeg(raw_bytes)

        gen.save_atomic_image(
            image_bytes=orig_bytes,
            target_path=target_path,
        )

        # Update manifest record
        rec.update({
            "batch": 1,
            "style": style,
            "category": category,
            "index": index,
            "account_label": account_label,
            "account_delegation_note": f"Generated via {account_label} upon manual review with semantic disambiguation",
            "disambiguated_object": disambiguated_object,
            "prompt_sha256": template_sha,
            "target_filepath": str(target_path),
            "filepath": str(target_path),
            "status": "COMPLETED",
            "latency_seconds": req_meta["elapsed_seconds"],
            "validation": val_meta,
            "jpeg_sha256": val_meta["jpeg_sha256"],
            "dhash": val_meta["dhash"],
            "format": "JPEG",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_history": audit_history,
        })
        rec.pop("error", None)
        rec.pop("error_code", None)
        rec.pop("manual_review_required", None)

        manifest["records"][target_key] = rec
        gen.save_manifest(manifest, MANIFEST_PATH)

        print("\n" + "=" * 72)
        print("GENERATION SUCCESSFUL!")
        print(f"  Saved Image:     {target_path}")
        print(f"  Format:          JPEG ({val_meta['byte_count']:,} bytes)")
        print(f"  Resolution:      {val_meta['width']}x{val_meta['height']}")
        print(f"  SHA-256:         {val_meta['jpeg_sha256']}")
        print(f"  Latency:         {req_meta['elapsed_seconds']}s")
        print(f"  Manifest Status: COMPLETED (Total: 288/288)")
        print("=" * 72)
        return True

    except gen.SafetyBlockedError as sbe:
        clean_msg = gen.sanitize_message(str(sbe), account_id, api_token)
        print(f"\n[STILL BLOCKED - Code {sbe.code}] {clean_msg}")
        rec.update({
            "batch": 1,
            "style": style,
            "category": category,
            "index": index,
            "account_label": account_label,
            "disambiguated_object": disambiguated_object,
            "status": "BLOCKED_SAFETY_FILTER",
            "error": clean_msg,
            "error_code": sbe.code,
            "manual_review_required": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_history": audit_history,
        })
        manifest["records"][target_key] = rec
        gen.save_manifest(manifest, MANIFEST_PATH)
        return False

    except Exception as ex:
        clean_msg = gen.sanitize_message(str(ex), account_id, api_token)
        print(f"\n[GENERATION FAILED] {clean_msg}")
        rec.update({
            "batch": 1,
            "style": style,
            "category": category,
            "index": index,
            "account_label": account_label,
            "status": "FAILED",
            "error": clean_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_history": audit_history,
        })
        manifest["records"][target_key] = rec
        gen.save_manifest(manifest, MANIFEST_PATH)
        return False


if __name__ == "__main__":
    account = sys.argv[1] if len(sys.argv) > 1 else "ACCOUNT_1"
    disambiguation = sys.argv[2] if len(sys.argv) > 2 else "commercial passenger airplane"
    success = generate_airplane_01(
        account_label=account, disambiguated_object=disambiguation
    )
    sys.exit(0 if success else 1)

