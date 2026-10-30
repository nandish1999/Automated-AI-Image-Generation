"""recover_blocked_tasks.py

Controlled Recovery Procedure for Safety-Blocked Tasks
Cloudflare Workers AI — 288 Image Research Dataset
Model: @cf/black-forest-labs/flux-2-klein-4b

Architecture & Safety Guarantees:
1. Strictly targets ONLY the tasks marked as BLOCKED_SAFETY_FILTER in dataset_manifest.json.
2. Excludes and protects all COMPLETED images (never overwritten or re-requested).
3. Preserves approved master prompt templates verbatim (photorealistic.txt and sketch.txt).
4. Preserves batch boundaries and dedicated account assignments (ACCOUNT_1, 2, 3, 5).
5. Preserves full audit history (archives original Code 3030 rejection timestamps and messages).
6. Requires explicit manual confirmation flag (--confirm-recovery) before execution.
7. Supports --dry-run and --mock modes for offline testing with zero live API calls.
"""

import sys
import json
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

import generate_images as gen

WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = WORKSPACE_DIR / "dataset_manifest.json"
DEFAULT_CREDENTIALS_PATH = WORKSPACE_DIR / "credentials.txt"


def identify_blocked_tasks(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> List[Dict[str, Any]]:
    """
    Scans the dataset manifest and extracts only the tasks with status 'BLOCKED_SAFETY_FILTER'.
    Asserts that no completed image is included.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", {})
    blocked_tasks = []

    for key, rec in records.items():
        status = rec.get("status")
        if status == "BLOCKED_SAFETY_FILTER":
            target_path = Path(rec.get("target_filepath"))
            blocked_tasks.append({
                "key": key,
                "batch": rec.get("batch"),
                "style": rec.get("style"),
                "category": rec.get("category"),
                "index": rec.get("index"),
                "account_label": rec.get("account_label"),
                "target_path": target_path,
                "original_error": rec.get("error"),
                "original_error_code": rec.get("error_code", 3030),
                "original_timestamp": rec.get("timestamp"),
            })

    # Sort deterministically by batch and category
    blocked_tasks.sort(key=lambda x: (x["batch"], x["category"], x["index"]))
    return blocked_tasks


def execute_controlled_recovery(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    credentials_path: Path = DEFAULT_CREDENTIALS_PATH,
    mock: bool = False,
    dry_run: bool = False,
    confirmed: bool = False,
) -> Dict[str, Any]:
    """
    Executes the controlled recovery plan for the identified blocked tasks.
    Guarantees that completed images are untouched, audit history is appended,
    and zero live API requests occur unless confirmed.
    """
    blocked_tasks = identify_blocked_tasks(manifest_path)
    manifest = gen.load_manifest(manifest_path)

    print("=" * 72)
    print("CONTROLLED RECOVERY: SAFETY-BLOCKED TASK RESOLUTION")
    print(f"Candidates Identified: {len(blocked_tasks)} tasks")
    print(f"Mode: {'DRY RUN' if dry_run else ('MOCK TEST (Offline)' if mock else 'LIVE RESOLUTION')}")
    print("=" * 72)

    if not blocked_tasks:
        print("No safety-blocked tasks found in manifest. Dataset is complete or clean.")
        return manifest

    if dry_run or not confirmed:
        print("\n[RECOVERY PLAN SUMMARY]")
        print(f"  Total Candidates: {len(blocked_tasks)} tasks")
        print("  Status: AWAITING EXPLICIT CONFIRMATION" if not confirmed else "  Status: DRY RUN COMPLETE")
        print("\n  Candidate Tasks to Recover:")
        for idx, t in enumerate(blocked_tasks, 1):
            print(f"    {idx:2d}. [{t['key']:30}] Batch {t['batch']} ({t['account_label']}) -> {t['target_path'].name}")

        if not confirmed and not dry_run:
            print("\n  [SAFETY NOTICE] Execution halted because --confirm-recovery was not passed.")
            print("  To simulate recovery offline, pass: --mock --confirm-recovery")
        return manifest

    # Load templates and verify checksums
    templates = {
        "photorealistic": gen.load_prompt_template("photorealistic"),
        "sketch": gen.load_prompt_template("sketch"),
    }

    recovered_count = 0
    still_blocked_count = 0

    for idx, t in enumerate(blocked_tasks, 1):
        key = t["key"]
        target_path: Path = t["target_path"]
        batch = t["batch"]
        style = t["style"]
        cat = t["category"]
        acc_label = t["account_label"]

        # 1. Existing-file protection: ensure we never touch completed images
        if target_path.exists() and target_path.stat().st_size > 0:
            print(f"  [{idx:02d}/{len(blocked_tasks)}] Skipping: valid file already on disk for {t['target_path'].name}")
            continue

        print(f"  [{idx:02d}/{len(blocked_tasks)}] Resubmitting candidate: {key} via {acc_label}...")

        # Load designated batch credentials strictly from credentials.txt
        account_id, api_token, _ = ("", "", acc_label)
        if not mock:
            account_id, api_token, _ = gen.load_credentials_for_batch(
                batch, credentials_file=credentials_path
            )

        template_text, template_sha = templates[style]
        prompt = gen.build_prompt_for_object(template_text, cat)

        # Retrieve prior record to preserve audit history
        prior_record = manifest["records"].get(key, {})
        audit_history = prior_record.get("audit_history", [])
        # Record the prior block in audit history if not already present
        if prior_record.get("status") == "BLOCKED_SAFETY_FILTER":
            audit_history.append({
                "attempt_timestamp": prior_record.get("timestamp"),
                "status": "BLOCKED_SAFETY_FILTER",
                "error_code": prior_record.get("error_code", 3030),
                "error_message": prior_record.get("error"),
                "account_label": acc_label,
            })

        new_record: Dict[str, Any] = {
            "batch": batch,
            "style": style,
            "category": cat,
            "index": t["index"],
            "account_label": acc_label,
            "prompt_sha256": template_sha,
            "target_filepath": str(target_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audit_history": audit_history,
        }

        try:
            raw_bytes, req_meta = gen.generate_single_image_with_resilience(
                account_id=account_id,
                api_token=api_token,
                prompt=prompt,
                mock=mock,
            )

            orig_bytes, val_meta = gen.validate_original_jpeg(raw_bytes)

            gen.save_atomic_image(
                image_bytes=orig_bytes,
                target_path=target_path,
            )

            new_record["status"] = "COMPLETED"
            new_record["latency_seconds"] = req_meta["elapsed_seconds"]
            new_record["validation"] = val_meta
            new_record["filepath"] = str(target_path)
            new_record["jpeg_sha256"] = val_meta["jpeg_sha256"]
            new_record["dhash"] = val_meta["dhash"]
            new_record["format"] = "JPEG"
            new_record["recovery_note"] = "Successfully recovered via controlled manual protocol"

            manifest["records"][key] = new_record
            gen.save_manifest(manifest, manifest_path)
            recovered_count += 1
            print(f"     [RECOVERED OK] Saved original JPEG ({val_meta['byte_count']:,} B)")

        except gen.SafetyBlockedError as sbe:
            clean_msg = gen.sanitize_message(str(sbe), account_id, api_token)
            print(f"     [RE-BLOCKED by Safety Filter] Code {sbe.code}: {clean_msg}")
            new_record["status"] = "BLOCKED_SAFETY_FILTER"
            new_record["error"] = clean_msg
            new_record["error_code"] = sbe.code
            new_record["manual_review_required"] = True
            manifest["records"][key] = new_record
            gen.save_manifest(manifest, manifest_path)
            still_blocked_count += 1

        except Exception as ex:
            clean_msg = gen.sanitize_message(str(ex), account_id, api_token)
            print(f"     [RECOVERY FAILED] {clean_msg}")
            new_record["status"] = "FAILED"
            new_record["error"] = clean_msg
            manifest["records"][key] = new_record
            gen.save_manifest(manifest, manifest_path)

    gen.save_manifest(manifest, manifest_path)
    print("\n" + "=" * 72)
    print("CONTROLLED RECOVERY SUMMARY")
    print(f"  Successfully Recovered: {recovered_count}")
    print(f"  Still Blocked:          {still_blocked_count}")
    print("=" * 72)
    return manifest


if __name__ == "__main__":
    args = sys.argv[1:]
    is_dry_run = "--dry-run" in args
    is_mock = "--mock" in args
    is_confirmed = "--confirm-recovery" in args

    execute_controlled_recovery(
        mock=is_mock,
        dry_run=is_dry_run,
        confirmed=is_confirmed,
    )

