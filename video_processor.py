"""
Video processing pipeline for HindiClip uploads.

This module wraps the original Whisper + MoviePy script in a reusable function
that can be triggered by the web backend whenever a creator uploads a file.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import subprocess
import tempfile

import whisper
from moviepy.editor import (
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)


logger = logging.getLogger(__name__)

_MODEL = None
HOOK_KEYWORDS = [
    "वाह",
    "अद्भुत",
    "शानदार",
    "मजेदार",
    "हंसी",
    "सुपर",
    "बेस्ट",
    "क्या बात",
    "कमाल",
    "यार",
    "भाई",
    "पागल",
    "लिट",
    "फायर",
    "धांसू",
    "देखो",
    "सुनो",
    "हाहा",
    "😂",
    "🔥",
    "💯",
    "wow",
    "amazing",
    "crazy",
    "funny",
    "haha",
    "bro",
    "insane",
    "lit",
    "fire",
]


@dataclass
class ClipSelection:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _load_model(model_size: str = "base"):
    global _MODEL
    if _MODEL is None:
        logger.info("Loading Whisper model '%s'...", model_size)
        _MODEL = whisper.load_model(model_size)
    return _MODEL


def _is_hook(text: str) -> bool:
    t = text.lower()
    if "!" in text or "?" in text:
        return True
    return any(keyword in t for keyword in HOOK_KEYWORDS)


def _get_candidate_segments(
    segments: Sequence[dict], duration: float
) -> List[ClipSelection]:
    candidates: List[ClipSelection] = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        if _is_hook(text) or len(text.split()) > 10:
            start = max(0.0, float(segment.get("start", 0)) - 2)
            end = min(duration, float(segment.get("end", 0)) + 2)
            if end - start >= 10:
                candidates.append(ClipSelection(start=start, end=end, text=text))
    if candidates:
        return candidates
    # fallback evenly spaced clips (ensure at least 3 segments)
    if not duration:
        return []
    chunk_count = max(3, int(duration // 20) or 3)
    step = duration / chunk_count
    clips = []
    for i in range(chunk_count):
        start = i * step
        end = min(duration, (i + 1) * step + 2)
        if end - start >= 8:
            clips.append(ClipSelection(start=start, end=end, text="Best Moment"))
    return clips


def _select_final_segments(
    candidates: Sequence[ClipSelection],
    duration: float,
    min_clip: float = 12.0,
    max_clip: float = 45.0,
    max_total: float = 90.0,
    target_window: float = 30.0,
) -> List[ClipSelection]:
    def _score_candidate(candidate: ClipSelection) -> float:
        hook_bonus = 2.0 if _is_hook(candidate.text) else 0.0
        length_score = min(1.5, candidate.duration / max_clip)
        early_bonus = max(0.0, 1.0 - candidate.start / max(duration, 1))
        return hook_bonus + length_score + early_bonus

    sorted_candidates = sorted(candidates, key=_score_candidate, reverse=True)
    selected: List[ClipSelection] = []
    total = 0.0
    for candidate in sorted_candidates:
        clip_duration = min(candidate.duration, max_clip)
        start, end = candidate.start, candidate.end

        if clip_duration > target_window:
            center = (candidate.start + candidate.end) / 2
            half_window = target_window / 2
            start = max(0.0, center - half_window)
            end = min(duration, start + target_window)
            clip_duration = end - start

        if min_clip <= clip_duration <= max_clip and total + clip_duration <= max_total:
            selected.append(ClipSelection(start=start, end=end, text=candidate.text))
            total += clip_duration
        if total >= max_total * 0.9:
            break
    return selected


def _fit_reel_aspect(base_clip: VideoFileClip, aspect: tuple[int, int]) -> VideoFileClip:
    """
    Crop the clip to the desired reel aspect ratio without downscaling.
    """
    target_ratio = aspect[0] / aspect[1]
    ratio = base_clip.w / base_clip.h
    if abs(ratio - target_ratio) < 0.01:
        return base_clip

    if ratio > target_ratio:
        # Too wide: crop the sides to center the subject.
        target_width = int(round(base_clip.h * target_ratio)) // 2 * 2
        x1 = max(0, (base_clip.w - target_width) // 2)
        x2 = x1 + target_width
        return base_clip.crop(x1=x1, x2=x2)

    # Too tall: crop top/bottom.
    target_height = int(round(base_clip.w / target_ratio)) // 2 * 2
    y1 = max(0, (base_clip.h - target_height) // 2)
    y2 = y1 + target_height
    return base_clip.crop(y1=y1, y2=y2)


def _create_caption_clip(text: str, base_clip: VideoFileClip) -> TextClip:
    font = "NotoSansDevanagari-Regular.ttf"
    if not any(ord(ch) > 127 for ch in text):
        font = "Arial-Bold"
    # Scale typography to video height so captions remain readable on low-res sources.
    base_height = max(1, base_clip.h)
    fontsize = max(28, min(96, int(base_height * 0.09)))
    stroke_width = max(2, min(6, int(fontsize * 0.08)))
    text_width = max(200, base_clip.w - 40)  # avoid clipping on narrow videos
    return (
        TextClip(
            text,
            fontsize=fontsize,
            color="white",
            stroke_color="black",
            stroke_width=stroke_width,
            font=font,
            size=(text_width, None),
            method="caption",
            align="center",
        )
        .set_position(("center", 0.78), relative=True)
        .set_duration(base_clip.duration)
        .crossfadein(0.5)
    )


def _clip_filename(job_id: str, index: int) -> str:
    return f"{job_id}_clip_{index}.mp4"


def _compilation_filename(job_id: str) -> str:
    return f"{job_id}_compilation.mp4"


def _probe_audio(input_video: Path) -> tuple[bool, float]:
    """
    Quickly probe for an audio stream and its duration so we can fail fast on silent/broken audio.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type,duration",
            "-of",
            "default=nw=1:nk=1",
            str(input_video),
        ]
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True
        )
        output = result.stdout.strip().splitlines()
        if not output:
            return False, 0.0
        durations = []
        for line in output:
            try:
                durations.append(float(line))
            except ValueError:
                continue
        duration = max(durations) if durations else 0.0
        return True, duration
    except Exception:  # noqa: BLE001
        # If probe fails, assume audio exists to avoid false negatives.
        logger.warning("Could not probe audio stream; proceeding with audio enabled.")
        return True, 0.0


def _extract_audio_snippet(input_video: Path, seconds: float | None) -> Path | None:
    """
    Extract only the first N seconds of audio to speed up transcription.
    Returns a temporary wav file path that caller must clean up.
    """
    if not seconds or seconds <= 0:
        return None
    tmp_file = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [
        "ffmpeg",
        "-y",
        "-t",
        str(seconds),
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(tmp_file),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp_file
    except Exception as exc:  # noqa: BLE001
        try:
            tmp_file.unlink(missing_ok=True)
        except OSError:
            pass
        logger.warning("Failed to extract audio snippet, falling back to full audio: %s", exc)
        return None


def process_video(
    input_video: Path,
    output_dir: Path,
    model_size: str = "base",
    target_aspect: tuple[int, int] = (9, 16),
    min_clip: float = 12.0,
    max_clip: float = 45.0,
    target_window: float = 30.0,
    fast_preview: bool = False,
    sample_first_seconds: float | None = None,
) -> dict:
    """
    Run the Whisper + MoviePy pipeline for a single uploaded video.

    Args:
        input_video: Path to the uploaded MP4 file.
        output_dir: Directory where generated clips and compilation are written.
        model_size: Whisper model size label (defaults to "base").
        target_aspect: Desired aspect ratio (w, h) for reels; crops without downscaling.
        min_clip: Minimum duration (seconds) for any individual clip.
        max_clip: Maximum duration (seconds) for any individual clip.
        target_window: Preferred reel length; longer clips are center-trimmed to this.
        fast_preview: If True, keep processing minimal (single short clip, faster encode).
        sample_first_seconds: If set, only transcribe the first N seconds to speed up.

    Returns:
        dict: Metadata describing created clips and their download URLs.
    """
    input_video = Path(input_video)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    model = _load_model(model_size)
    logger.info("Processing %s (job %s)", input_video.name, job_id)

    has_audio, audio_duration = _probe_audio(input_video)
    if not has_audio:
        raise ValueError(
            "No audio stream found in the uploaded video; please upload a video with sound."
        )
    if audio_duration <= 0.05:
        raise ValueError(
            "Audio stream appears to be empty or corrupt; please upload a video with audible sound."
        )

    video = VideoFileClip(str(input_video), audio=True)
    source_fps = max(1, min(60, getattr(video, "fps", 30) or 30))
    export_kwargs = {
        "codec": "libx264",
        "audio_codec": "aac",
        "threads": 8,
        "preset": "slow",
        "ffmpeg_params": ["-crf", "16", "-movflags", "+faststart"],
    }
    if fast_preview:
        # Faster encode, acceptable quality for a quick preview.
        export_kwargs["preset"] = "fast"
        export_kwargs["ffmpeg_params"] = ["-crf", "22", "-movflags", "+faststart"]
    final_clip_paths: List[Path] = []
    clips_metadata = []
    compilation_metadata = None

    try:
        duration = float(video.duration)
        transcript_temp = None
        transcript_target = str(input_video)
        transcription_kwargs = {"word_timestamps": True}
        if fast_preview and sample_first_seconds:
            transcript_temp = _extract_audio_snippet(input_video, sample_first_seconds)
            if transcript_temp:
                transcript_target = str(transcript_temp)
        try:
            result = model.transcribe(transcript_target, **transcription_kwargs)
        except Exception as exc:  # noqa: BLE001
            if "cannot reshape tensor of 0 elements" in str(exc):
                raise ValueError(
                    "Transcription failed because the audio stream is empty. Please upload a video with audible sound."
                ) from exc
            raise
        finally:
            if transcript_temp:
                try:
                    Path(transcript_temp).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to clean up temp audio snippet %s", transcript_temp)
        segments = result.get("segments", [])
        candidates = _get_candidate_segments(segments, duration)
        selected = _select_final_segments(
            candidates,
            duration=duration,
            min_clip=min_clip,
            max_clip=max_clip,
            target_window=target_window,
        )
        if fast_preview and selected:
            # Keep only the best single clip for faster turnaround.
            selected = [selected[0]]

        if not selected:
            logger.warning("No valid segments found for %s", input_video)
            return {
                "jobId": job_id,
                "videoDuration": duration,
                "clips": [],
                "compilation": None,
                "segments": [],
                "message": "No clips created",
            }

        for index, selection in enumerate(selected, start=1):
            subclip = video.subclip(selection.start, selection.end)
            reel_clip = _fit_reel_aspect(subclip, target_aspect)
            caption = _create_caption_clip(selection.text, reel_clip)
            final = CompositeVideoClip([reel_clip, caption])
            clip_filename = _clip_filename(job_id, index)
            target_path = output_dir / clip_filename
            final.write_videofile(
                str(target_path),
                fps=source_fps,
                **export_kwargs,
                verbose=False,
                logger=None,
            )
            final.close()
            if reel_clip is not subclip:
                reel_clip.close()
            subclip.close()
            caption.close()
            final_clip_paths.append(target_path)
            clip_info = {
                "file": clip_filename,
                "url": f"/outputs/{clip_filename}",
                "start": selection.start,
                "end": selection.end,
                "duration": selection.duration,
                "text": selection.text,
            }
            clips_metadata.append(clip_info)

        if final_clip_paths and not fast_preview:
            compilation_name = _compilation_filename(job_id)
            compilation_path = output_dir / compilation_name
            composite_clips = [VideoFileClip(str(path)) for path in final_clip_paths]
            compilation = concatenate_videoclips(composite_clips, method="compose")
            compilation.write_videofile(
                str(compilation_path),
                fps=source_fps,
                **export_kwargs,
                verbose=False,
                logger=None,
            )
            compilation.close()
            for clip in composite_clips:
                clip.close()
            compilation_metadata = {
                "file": compilation_name,
                "url": f"/outputs/{compilation_name}",
                "duration": sum(item["duration"] for item in clips_metadata),
            }

        message = f"Created {len(clips_metadata)} clips"
        if compilation_metadata:
            message += " + compilation"

        return {
            "jobId": job_id,
            "videoDuration": duration,
            "clips": clips_metadata,
            "compilation": compilation_metadata,
            "segments": [
                {"start": clip.start, "end": clip.end, "text": clip.text}
                for clip in selected
            ],
            "message": message,
        }
    finally:
        video.close()
