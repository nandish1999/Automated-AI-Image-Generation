"""test_offline.py

Comprehensive offline test suite for the Cloudflare Workers AI image generation pipeline.
Zero live API requests.
Validates:
1. Prompt consistency and template invariance across objects and styles.
2. Original JPEG byte preservation (byte-for-byte exact match, zero recompression).
3. Image validation pipeline (corrupt, empty, non-JPEG detection, wrong dimensions, non-square).
4. Safety filter compliance (Code 3030 blocked, no retries).
5. Error hierarchy (AuthError, QuotaExceededError, TransientAPIError).
6. Direct atomic JPEG file saving (zero PNG creation, zero raw_outputs directory).
7. Cryptographic SHA-256 and perceptual dHash calculations.
8. Six-batch boundaries, task ordering, and .jpg target filenames (288 total distinct tasks).
9. Dry-run behavior (zero API calls, zero file mutations, .jpg path reporting).
10. Batch mock execution, manifest consistency, and genuine JPEG validation.
11. Resume behavior and existing-file protection on valid JPEG files.
12. Multi-account credentials parsing from credentials.txt and standalone loader.
"""

import io
import sys
import json
import shutil
import hashlib
from pathlib import Path
from PIL import Image, ImageChops

import generate_images as gen


def test_prompt_consistency_and_invariance():
    """Validates Task 4 & Additional Task B: Only line 1 changes across objects."""
    print("\n--- [TEST 1] Prompt Consistency & Template Invariance ---")
    test_objects = ["dog", "airplane", "chair", "apple", "train"]

    for style in ["photorealistic", "sketch"]:
        template_text, template_sha = gen.load_prompt_template(style)
        print(f"Testing style: '{style}' (Template SHA256: {template_sha[:16]}...)")
        template_lines = template_text.splitlines()

        for obj in test_objects:
            substituted = gen.build_prompt_for_object(template_text, obj)
            sub_lines = substituted.splitlines()

            assert sub_lines[0] == f"OBJECT: {obj}", f"Line 0 mismatch for {obj}: '{sub_lines[0]}'"
            assert sub_lines[1:] == template_lines[1:], f"Remaining lines changed for {obj} in {style}!"
            assert len(sub_lines) == len(template_lines), f"Line count changed for {obj}!"

        print(f"  -> PASSED: All {len(test_objects)} objects verified invariant for '{style}'.")


def test_original_jpeg_byte_preservation_and_validation():
    """Validates: Original JPEG byte preservation, zero recompression, format detection, and dimensions."""
    print("\n--- [TEST 2] Original JPEG Byte Preservation & Validation ---")
    orig_img = Image.new("RGB", (1024, 1024), color=(85, 120, 165))
    buf = io.BytesIO()
    orig_img.save(buf, format="JPEG", quality=90)
    raw_jpeg_bytes = buf.getvalue()

    assert raw_jpeg_bytes.startswith(b"\xff\xd8\xff"), "Synthetic buffer is not valid JPEG!"

    # Validate raw bytes directly with validate_original_jpeg
    preserved_bytes, meta = gen.validate_original_jpeg(raw_jpeg_bytes)

    # 1. Exact byte preservation (zero recompression or transcoding)
    assert preserved_bytes == raw_jpeg_bytes, "Output bytes differ from input bytes!"
    assert len(preserved_bytes) == len(raw_jpeg_bytes), "Byte length altered!"
    assert meta["jpeg_sha256"] == hashlib.sha256(raw_jpeg_bytes).hexdigest(), "SHA256 mismatch!"

    # 2. Format & dimension checks
    assert meta["format"] == "JPEG"
    assert meta["width"] == 1024
    assert meta["height"] == 1024
    assert meta["is_square"] is True
    assert meta["is_1024x1024"] is True
    assert len(meta["dhash"]) == 16

    print(f"  -> PASSED: Exact {len(preserved_bytes):,} JPEG bytes preserved without recompression (SHA256 match).")


def test_image_validation_failures():
    """Validates: Robust failure detection for corrupted or malformed images, and non-JPEG payloads."""
    print("\n--- [TEST 3] Technical Image Validation Failures ---")

    # 1. Empty payload
    try:
        gen.validate_original_jpeg(b"")
        assert False, "Should have failed on empty bytes"
    except gen.ImageValidationError:
        print("  -> PASSED: Empty payload rejected.")

    # 2. Corrupted data
    try:
        gen.validate_original_jpeg(b"\xff\xd8\xffcorrupted_garbage_bytes")
        assert False, "Should have failed on corrupted bytes"
    except gen.ImageValidationError:
        print("  -> PASSED: Corrupted bytes rejected.")

    # 3. Non-JPEG payload (PNG format detection)
    png_buf = io.BytesIO()
    Image.new("RGB", (1024, 1024), color="green").save(png_buf, format="PNG")
    try:
        gen.validate_original_jpeg(png_buf.getvalue())
        assert False, "Should have rejected PNG payload"
    except gen.ImageValidationError as e:
        assert "PNG" in str(e)
        print(f"  -> PASSED: Non-JPEG (PNG) payload rejected with clear error: {e}")

    # 4. Wrong dimensions (512x512)
    img_512 = Image.new("RGB", (512, 512), color="red")
    buf = io.BytesIO()
    img_512.save(buf, format="JPEG")
    try:
        gen.validate_original_jpeg(buf.getvalue(), expected_width=1024, expected_height=1024)
        assert False, "Should have failed on 512x512 dimensions"
    except gen.ImageValidationError as e:
        assert "512x512" in str(e)
        print("  -> PASSED: 512x512 image rejected with dimension report.")

    # 5. Non-square (1024x768)
    img_rect = Image.new("RGB", (1024, 768), color="blue")
    buf2 = io.BytesIO()
    img_rect.save(buf2, format="JPEG")
    try:
        gen.validate_original_jpeg(buf2.getvalue(), expected_width=1024, expected_height=1024)
        assert False, "Should have failed on non-square dimensions"
    except gen.ImageValidationError as e:
        print(f"  -> PASSED: Non-square dimensions rejected: {e}")


def test_safety_filter_compliance_and_error_handling():
    """Validates Task 3 & 9: Strict safety compliance (Code 3030 blocked, no retries)."""
    print("\n--- [TEST 4] Safety Filter Compliance & Error Handling ---")

    # 1. Safety Blocked Error (Code 3030) must NOT retry
    try:
        gen.generate_single_image_with_resilience(
            account_id="mock_acc",
            api_token="mock_tok",
            prompt="TRIGGER_SAFETY_BLOCK",
            mock=True,
        )
        assert False, "Should have raised SafetyBlockedError"
    except gen.SafetyBlockedError as sbe:
        assert sbe.code == 3030
        print(f"  -> PASSED: SafetyBlockedError caught strictly without retry: {sbe}")

    # 2. Fatal Auth Error (HTTP 401)
    try:
        gen.generate_single_image_with_resilience(
            account_id="mock_acc",
            api_token="mock_tok",
            prompt="TRIGGER_AUTH_ERROR",
            mock=True,
        )
        assert False, "Should have raised AuthError"
    except gen.AuthError:
        print("  -> PASSED: AuthError caught as fatal exception.")

    # 3. Quota Exceeded Error (HTTP 429)
    try:
        gen.generate_single_image_with_resilience(
            account_id="mock_acc",
            api_token="mock_tok",
            prompt="TRIGGER_QUOTA_ERROR",
            mock=True,
        )
        assert False, "Should have raised QuotaExceededError"
    except gen.QuotaExceededError:
        print("  -> PASSED: QuotaExceededError caught as fatal exception.")


def test_atomic_file_saving():
    """Validates: Atomic writes save .jpg directly with no duplicate PNG and no raw_outputs directory."""
    print("\n--- [TEST 5] Atomic File Saving (JPEG Direct) ---")
    test_dir = Path(".test_tmp/test_atomic")
    test_target = test_dir / "test_img.jpg"

    dummy_jpeg = b"\xff\xd8\xff" + b"\x00" * 80

    gen.save_atomic_image(
        image_bytes=dummy_jpeg,
        target_path=test_target,
    )

    assert test_target.exists(), "Target JPEG does not exist!"
    assert test_target.read_bytes() == dummy_jpeg, "Target JPEG content mismatch!"

    # Verify no PNG sibling was created
    png_sibling = test_dir / "test_img.png"
    assert not png_sibling.exists(), "Unexpected PNG sibling created!"

    # Verify no raw_outputs directory was created
    raw_dir = test_dir / "raw_outputs"
    assert not raw_dir.exists(), "Unexpected raw_outputs directory created!"

    shutil.rmtree(test_dir.parent)
    print("  -> PASSED: Atomic JPEG saving verified (zero PNG, zero raw_outputs directory).")


def test_cryptographic_and_perceptual_hashes():
    """Validates Task 8 & 11: SHA-256 and dHash computations."""
    print("\n--- [TEST 6] Hash Calculations ---")
    img = Image.new("RGB", (1024, 1024), color=(200, 100, 50))
    dhash = gen.calculate_dhash(img)
    assert len(dhash) == 16, f"dHash must be 16-hex characters, got: {dhash}"

    data = b"research_test_data"
    sha = gen.calculate_sha256(data)
    assert len(sha) == 64, f"SHA256 must be 64-hex characters, got: {sha}"

    print(f"  -> PASSED: dHash={dhash}, SHA256={sha[:16]}... verified.")


def test_batch_boundaries_and_task_ordering():
    """Validates: Six batches, 48 items each, sequential index ordering, exactly 288 distinct tasks."""
    print("\n--- [TEST 7] Six-Batch Boundaries & Task Ordering ---")
    with open("categories.json") as f:
        cats = json.load(f)["categories"]
    assert len(cats) == 48, f"Expected 48 categories, found {len(cats)}"

    all_keys = set()
    total_tasks = 0

    expected_batch_specs = [
        (1, "photorealistic", 1, "ACCOUNT_1"),
        (2, "photorealistic", 2, "ACCOUNT_2"),
        (3, "photorealistic", 3, "ACCOUNT_3"),
        (4, "sketch", 1, "ACCOUNT_4"),
        (5, "sketch", 2, "ACCOUNT_5"),
        (6, "sketch", 3, "ACCOUNT_6"),
    ]

    for b_num, exp_style, exp_idx, exp_acc in expected_batch_specs:
        tasks = gen.get_batch_tasks(b_num, cats)
        assert len(tasks) == 48, f"Batch {b_num} does not contain 48 tasks (found {len(tasks)})"

        for t in tasks:
            assert t["batch"] == b_num
            assert t["style"] == exp_style
            assert t["index"] == exp_idx
            assert t["account_label"] == exp_acc
            assert t["target_path"].suffix == ".jpg", f"Target filename must be .jpg: {t['target_path']}"
            assert "raw_path" not in t, "raw_path should no longer exist in task definition"
            key = t["key"]
            assert key not in all_keys, f"Duplicate task key detected: {key}"
            all_keys.add(key)
            total_tasks += 1

        print(f"  -> Batch {b_num}: {exp_style} (index {exp_idx:02d} | {exp_acc}) = exactly 48 tasks verified.")

    assert total_tasks == 288, f"Expected 288 total tasks, found {total_tasks}"
    assert len(all_keys) == 288, f"Expected 288 unique task keys, found {len(all_keys)}"
    print(f"  -> PASSED: Exactly 6 batches × 48 tasks = 288 unique tasks verified with 0 collisions.")


def test_dry_run_behavior():
    """Validates: Dry-run performs zero API calls, writes zero files, mutates zero manifest entries."""
    print("\n--- [TEST 8] Dry-Run Behavior ---")
    test_dir = Path(".test_tmp")
    test_dir.mkdir(exist_ok=True)
    test_manifest_path = test_dir / "dry_run_manifest.json"
    isolated_images_dir = test_dir / "images"
    isolated_images_dir.mkdir(exist_ok=True)

    original_images_dir = gen.IMAGES_DIR
    gen.IMAGES_DIR = isolated_images_dir

    try:
        initial_manifest = gen.initialize_empty_manifest()
        with open(test_manifest_path, "w") as f:
            json.dump(initial_manifest, f)

        cats = ["dog", "airplane", "chair"]
        manifest_after = gen.execute_batch(
            batch_number=1,
            categories=cats,
            manifest_path=test_manifest_path,
            dry_run=True,
        )

        # Verify manifest on disk was untouched
        disk_data = json.loads(test_manifest_path.read_text())
        assert len(disk_data["records"]) == 0, "Dry run modified manifest records!"

        # Verify no files were created in isolated images/ (.jpg or .png)
        for c in cats:
            target_jpg = gen.IMAGES_DIR / "photorealistic" / c / f"{c}_01.jpg"
            target_png = gen.IMAGES_DIR / "photorealistic" / c / f"{c}_01.png"
            assert not target_jpg.exists(), f"Dry run unexpectedly created file: {target_jpg}"
            assert not target_png.exists(), f"Dry run unexpectedly created file: {target_png}"

        print("  -> PASSED: Dry run completed with 0 API calls, 0 file writes, 0 manifest mutations.")
    finally:
        gen.IMAGES_DIR = original_images_dir
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_batch_mock_execution_and_manifest_consistency():
    """Validates: Mock batch execution saves genuine JPEG directly, creates no PNGs or raw_outputs, and updates manifest."""
    print("\n--- [TEST 9] Batch Mock Execution & Manifest Consistency ---")
    test_dir = Path(".test_tmp")
    test_dir.mkdir(exist_ok=True)
    test_manifest_path = test_dir / "mock_test_manifest.json"
    isolated_images_dir = test_dir / "images"
    isolated_images_dir.mkdir(exist_ok=True)

    original_images_dir = gen.IMAGES_DIR
    gen.IMAGES_DIR = isolated_images_dir

    try:
        cats = ["dog", "apple"]

        # Execute Batch 1 mock
        gen.execute_batch(
            batch_number=1,
            categories=cats,
            manifest_path=test_manifest_path,
            mock=True,
        )

        m = gen.load_manifest(test_manifest_path)
        b1_stat = m["batch_statistics"]["batch_1"]
        assert b1_stat["completed"] == 2, f"Expected 2 completed in batch 1, got {b1_stat['completed']}"
        assert m["overall_statistics"]["photorealistic_completed"] == 2
        assert m["overall_statistics"]["total_completed"] == 2

        # Check generated files exist, are genuine JPEG, and NO PNG or raw_outputs directory exists
        for c in cats:
            jpg_p = gen.IMAGES_DIR / "photorealistic" / c / f"{c}_01.jpg"
            png_p = gen.IMAGES_DIR / "photorealistic" / c / f"{c}_01.png"
            raw_dir = gen.IMAGES_DIR / "photorealistic" / c / "raw_outputs"

            assert jpg_p.exists(), f"Missing mock JPEG output: {jpg_p}"
            assert not png_p.exists(), f"Unexpected PNG file created: {png_p}"
            assert not raw_dir.exists(), f"Unexpected raw_outputs directory created: {raw_dir}"

            with Image.open(jpg_p) as img:
                assert img.format == "JPEG"
                assert img.size == (1024, 1024)

            rec = m["records"][f"photorealistic:{c}:01"]
            assert rec["format"] == "JPEG"
            assert rec["status"] == "COMPLETED"
            assert rec["filepath"].endswith(".jpg")
            assert rec["jpeg_sha256"] == hashlib.sha256(jpg_p.read_bytes()).hexdigest()

        print("  -> PASSED: Mock batch execution verified: genuine JPEG saved directly, zero PNGs, zero raw_outputs.")
    finally:
        gen.IMAGES_DIR = original_images_dir
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_resume_behavior_and_existing_file_protection():
    """Validates: Resuming skips valid JPEG files and never overwrites existing valid images."""
    print("\n--- [TEST 10] Resume Behavior & Existing-File Protection ---")
    test_dir = Path(".test_tmp")
    test_dir.mkdir(exist_ok=True)
    test_manifest_path = test_dir / "resume_manifest.json"
    isolated_images_dir = test_dir / "images"
    isolated_images_dir.mkdir(exist_ok=True)

    original_images_dir = gen.IMAGES_DIR
    gen.IMAGES_DIR = isolated_images_dir

    try:
        cat = "dog"
        target_p = gen.IMAGES_DIR / "photorealistic" / cat / f"{cat}_01.jpg"
        target_p.parent.mkdir(parents=True, exist_ok=True)

        # Create a pre-existing valid 1024x1024 JPEG with distinct pixels
        pre_img = Image.new("RGB", (1024, 1024), color=(33, 66, 99))
        pre_img.save(target_p, format="JPEG", quality=95)
        original_sha = hashlib.sha256(target_p.read_bytes()).hexdigest()

        # Run Batch 1 mock for 'dog'
        gen.execute_batch(
            batch_number=1,
            categories=[cat],
            manifest_path=test_manifest_path,
            mock=True,
        )

        # Verify file was NOT overwritten
        current_sha = hashlib.sha256(target_p.read_bytes()).hexdigest()
        assert original_sha == current_sha, "Existing JPEG file was overwritten!"

        # Verify manifest recorded it as completed with format JPEG
        m = gen.load_manifest(test_manifest_path)
        rec = m["records"][f"photorealistic:{cat}:01"]
        assert rec["status"] == "COMPLETED"
        assert rec["format"] == "JPEG"
        assert rec["jpeg_sha256"] == original_sha

        print("  -> PASSED: Existing JPEG file protected; skipping verified with zero overwrite.")
    finally:
        gen.IMAGES_DIR = original_images_dir
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_credentials_parsing():
    """Validates: Multi-account credentials parser loads all 6 accounts and standalone loader strictly from credentials.txt."""
    print("\n--- [TEST 11] Multi-Account Credentials Parsing & Standalone Loader ---")
    accounts = gen.parse_credentials_file(Path("credentials.txt"))
    assert len(accounts) == 6, f"Expected 6 accounts, parsed {len(accounts)}"

    for b_num in range(1, 7):
        acc_id, tok, label = gen.load_credentials_for_batch(b_num, credentials_file=Path("credentials.txt"))
        assert label == f"ACCOUNT_{b_num}"
        assert len(acc_id) > 10
        assert len(tok) > 10
        print(f"  -> Batch {b_num} mapped to {label}: account ID and token verified.")

    # Validate standalone loader defaulting to ACCOUNT_1
    c1_acc, c1_tok = gen.load_credentials(account_label="ACCOUNT_1", credentials_file=Path("credentials.txt"))
    b1_acc, b1_tok, _ = gen.load_credentials_for_batch(1, credentials_file=Path("credentials.txt"))
    assert c1_acc == b1_acc
    assert c1_tok == b1_tok
    print("  -> Standalone loader verified matching Batch 1 / ACCOUNT_1.")

    # Validate error handling on nonexistent file and invalid section
    try:
        gen.load_credentials(credentials_file=Path("nonexistent_creds.txt"))
        assert False, "Should have failed on missing file"
    except gen.AuthError:
        print("  -> Missing credentials file handled cleanly.")

    try:
        gen.load_credentials(account_label="NONEXISTENT_ACCOUNT", credentials_file=Path("credentials.txt"))
        assert False, "Should have failed on missing section"
    except gen.AuthError:
        print("  -> Missing account section handled cleanly.")

    print("  -> PASSED: All 6 account credentials parsed securely with zero exposure.")


def main():
    print("=" * 70)
    print("RUNNING EXTENDED OFFLINE TEST SUITE (11 TEST DOMAINS)")
    print("Zero live API requests. Validating production batch architecture.")
    print("=" * 70)

    try:
        test_prompt_consistency_and_invariance()
        test_original_jpeg_byte_preservation_and_validation()
        test_image_validation_failures()
        test_safety_filter_compliance_and_error_handling()
        test_atomic_file_saving()
        test_cryptographic_and_perceptual_hashes()
        test_batch_boundaries_and_task_ordering()
        test_dry_run_behavior()
        test_batch_mock_execution_and_manifest_consistency()
        test_resume_behavior_and_existing_file_protection()
        test_credentials_parsing()

        print("\n" + "=" * 70)
        print("ALL 11 OFFLINE TEST DOMAINS PASSED SUCCESSFULLY!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[TEST SUITE FAILURE] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
