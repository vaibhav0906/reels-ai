# reels-ai

HindiClip landing page with a Python backend that runs your Whisper + MoviePy
processing pipeline the moment a creator uploads a file.

## Requirements

- Python 3.10+
- FFmpeg available on your `$PATH` (required by MoviePy)
- `pip install -r requirements.txt`
- (Optional) Razorpay checkout: set `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET` in your
  environment to enable `/api/razorpay/order` for the payment page.

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000), sign in, and upload a
video. The Flask server stores temporary uploads in `uploads/`, renders clips
to `outputs/`, and responds with download links once the script finishes.

The heavy lifting lives in `video_processor.py`, which encapsulates the original
Whisper + MoviePy script so the front-end can trigger processing over HTTP.
