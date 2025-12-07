"""
Flask application that serves the HindiClip front-end and processes uploads.
"""
from __future__ import annotations

import base64
import json
import logging
import uuid
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from urllib import error as urlerror
from urllib import request as urlrequest

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from video_processor import process_video


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB
PROCESS_TIMEOUT = int(os.environ.get("PROCESS_TIMEOUT_SECONDS", "240"))
DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")
RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")
RAZORPAY_ORDER_URL = "https://api.razorpay.com/v1/orders"

app = Flask(
    __name__,
    static_folder=str(BASE_DIR),
    static_url_path="",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _ensure_directories():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_ensure_directories()


def _create_razorpay_order(amount_rupees: float, plan: str, email: str | None = None) -> dict:
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise ValueError("Razorpay keys are not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET.")
    amount_paise = int(round(amount_rupees * 100))
    payload = {
        "amount": amount_paise,
        "currency": "INR",
        "receipt": f"hindiclip_{plan}_{uuid.uuid4().hex[:8]}",
        "payment_capture": 1,
        "notes": {
            "plan": plan,
            "email": email or "",
            "source": "hindiclip_checkout",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    auth_header = base64.b64encode(f"{RAZORPAY_KEY_ID}:{RAZORPAY_KEY_SECRET}".encode("utf-8")).decode("utf-8")
    req = urlrequest.Request(
        RAZORPAY_ORDER_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth_header}",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urlerror.HTTPError as exc:  # pragma: no cover - runtime only
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"Razorpay order creation failed: {exc.code} {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Razorpay order creation failed: {exc}") from exc


@app.post("/api/razorpay/order")
def create_razorpay_order_api():
    data = request.get_json(force=True, silent=True) or {}
    amount = float(data.get("amount") or 0)
    plan = str(data.get("plan") or "custom")
    email = data.get("email") or ""
    if amount <= 0:
        return jsonify({"error": "Invalid amount"}), 400
    try:
        order = _create_razorpay_order(amount, plan, email=email)
        order["key"] = RAZORPAY_KEY_ID
        order["amount_rupees"] = amount
        return jsonify(order)
    except ValueError as exc:
        logger.error("Razorpay not configured: %s", exc)
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create Razorpay order for %s", plan)
        return jsonify({"error": "Unable to create Razorpay order", "detail": str(exc)}), 500


@app.post("/api/process-video")
def handle_process_video():
    if "video" not in request.files:
        return jsonify({"error": "Missing 'video' file field."}), 400

    source_file = request.files["video"]
    if source_file.filename == "":
        return jsonify({"error": "Empty filename supplied."}), 400

    filename = secure_filename(source_file.filename)
    temp_name = f"{uuid.uuid4().hex}_{filename}"
    _ensure_directories()
    temp_path = UPLOAD_DIR / temp_name
    source_file.save(temp_path)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                process_video,
                temp_path,
                OUTPUT_DIR,
                # Preserve as much source quality as possible and crop for 9:16 reels.
                target_aspect=(9, 16),
                min_clip=8.0,
                max_clip=24.0,
                target_window=18.0,
                model_size=DEFAULT_MODEL_SIZE,
                fast_preview=True,
                sample_first_seconds=90.0,
            )
            try:
                result = future.result(timeout=PROCESS_TIMEOUT)
            except TimeoutError:
                future.cancel()
                msg = (
                    "Processing is still running in the background. "
                    "We’ll let you know once your reel is ready."
                )
                logger.warning("Job timed out for %s", filename)
                return (
                    jsonify(
                        {
                            "error": msg,
                            "status": "processing",
                            "timeoutSeconds": PROCESS_TIMEOUT,
                        }
                    ),
                    504,
                )
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Processing failed for %s", filename)
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temporary file %s", temp_path)


@app.get("/outputs/<path:filename>")
def download_output(filename: str):
    _ensure_directories()
    return send_from_directory(
        OUTPUT_DIR, filename, as_attachment=True, download_name=filename
    )


@app.get("/")
def serve_index():
    return app.send_static_file("index.html")


@app.get("/<path:path>")
def serve_static(path: str):
    target = BASE_DIR / path
    if target.is_file():
        return send_from_directory(app.static_folder, path)
    return app.send_static_file("index.html")


if __name__ == "__main__":
    _ensure_directories()
    port = int(os.environ.get("PORT") or os.environ.get("FLASK_RUN_PORT") or 5000)
    app.run(debug=True, port=port)
