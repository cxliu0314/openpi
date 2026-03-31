"""
LLM-based replanner for robotic task recovery.

This module is intentionally isolated for OpenPI evaluation.
It does not modify existing OpenPI serving/inference behavior.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_BASE_URL = "http://127.0.0.1:3000/v1"

VALID_PRIMITIVES = {
    "init",
    "reach",
    "grasp",
    "release",
    "move",
    "hold",
    "rotate",
    "flip",
    "push",
    "pull",
    "insert",
    "press",
    "strike",
}


def _log(log_fn: Optional[Callable[[str], None]], message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def frame_to_data_url(frame_rgb: np.ndarray, jpg_quality: int = 90) -> str:
    """Convert RGB frame to a JPEG data URL."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_replan_prompt(
    previous_subtask: Optional[Dict],
    current_subtask: Optional[Dict],
    task_description: str,
    available_objects: List[str],
) -> str:
    """Build a strict JSON-only replan prompt."""
    prev_instruction = previous_subtask.get("instruction", "None") if previous_subtask else "None"
    curr_instruction = (
        current_subtask.get("instruction", "Unknown")
        if current_subtask
        else "Task completed but behavior appears incorrect"
    )
    curr_primitive = current_subtask.get("primitive", "unknown") if current_subtask else "task_complete"
    objects_str = ", ".join(available_objects) if available_objects else "Unknown"

    return f"""You are a robotic task recovery planner for LIBERO manipulation.

## Task Description
{task_description}

## Current Situation
- Previous subtask: {prev_instruction}
- Current subtask (stuck): {curr_instruction}
- Current primitive: {curr_primitive}
- Available objects: {objects_str}

## Recovery Rules
1. If current primitive is "reach", always begin with init().
2. If grasp failed, plan init() -> reach(obj) -> grasp(obj).
3. If object dropped during move, re-grasp then continue move/release.
4. Keep plan short and executable in simulator.

## Allowed primitives
["init","reach","grasp","release","move","hold","rotate","flip","push","pull","insert","press","strike"]

## Required output format
Return ONLY a JSON array. No prose. No markdown.
Each item must be:
{{
  "primitive": "<allowed primitive>",
  "args": [...],
  "instruction": "<human-readable function-like string>"
}}
"""


def _get_api_settings() -> Tuple[str, str]:
    api_base = os.getenv("OPENPI_REPLAN_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
    api_key = os.getenv("OPENPI_REPLAN_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENPI_REPLAN_API_KEY is not set")
    return api_base, api_key


def call_llm_with_images(
    prompt: str,
    image_data_urls: List[str],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    """Call OpenAI-compatible chat completions with image inputs."""
    api_base, api_key = _get_api_settings()
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    content: List[Dict] = [{"type": "text", "text": prompt}]
    for idx, data_url in enumerate(image_data_urls):
        content.append({"type": "text", "text": f"Frame {idx}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    output = data["choices"][0]["message"]["content"]
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        texts = [x.get("text", "") for x in output if x.get("type") == "text"]
        return "\n".join(x for x in texts if x)
    return str(output)


def parse_recovery_plan(llm_response: str, log_fn: Optional[Callable[[str], None]] = None) -> List[Dict]:
    """Parse and validate recovery subtasks from model response."""
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", llm_response)
    if code_block:
        json_str = code_block.group(1).strip()
    else:
        json_match = re.search(r"\[[\s\S]*\]", llm_response)
        if not json_match:
            _log(log_fn, "[LLM Replan] No JSON array found in response")
            return []
        json_str = json_match.group(0).strip()

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        _log(log_fn, f"[LLM Replan] JSON parse error: {exc}")
        return []

    if not isinstance(parsed, list):
        _log(log_fn, "[LLM Replan] Parsed JSON is not a list")
        return []

    validated: List[Dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        primitive = item.get("primitive", "")
        if primitive not in VALID_PRIMITIVES:
            continue
        validated.append(
            {
                "primitive": primitive,
                "args": item.get("args", []),
                "instruction": item.get("instruction", f"{primitive}()"),
                **({"kwargs": item["kwargs"]} if "kwargs" in item else {}),
            }
        )
    return validated


def sample_recent_frames(
    replay_images: List[np.ndarray],
    num_frames: int = 3,
    skip_frames: int = 2,
) -> List[np.ndarray]:
    """Sample recent frames with configurable skipping."""
    if not replay_images:
        return []
    total_needed = num_frames + (num_frames - 1) * skip_frames
    if len(replay_images) < total_needed:
        step = max(1, len(replay_images) // max(1, num_frames))
        indices = list(range(0, len(replay_images), step))[:num_frames]
    else:
        start = len(replay_images) - total_needed
        indices = list(range(start, len(replay_images), skip_frames + 1))[:num_frames]
    return [replay_images[i] for i in indices]


def get_recovery_plan(
    replay_images: List[np.ndarray],
    previous_subtask: Optional[Dict],
    current_subtask: Optional[Dict],
    task_description: str,
    available_objects: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    log_fn: Optional[Callable[[str], None]] = None,
    max_retries: int = 3,
) -> Tuple[List[Dict], bool]:
    """Get an LLM recovery plan with retry on parse/API failures."""
    sampled_frames = sample_recent_frames(replay_images, num_frames=3, skip_frames=2)
    if len(sampled_frames) < 2:
        _log(log_fn, "[LLM Replan] Not enough frames for replanning")
        return [], False

    try:
        image_urls = [frame_to_data_url(frame) for frame in sampled_frames]
    except Exception as exc:  # noqa: BLE001
        _log(log_fn, f"[LLM Replan] Frame encoding failed: {exc}")
        return [], False

    prompt = build_replan_prompt(
        previous_subtask=previous_subtask,
        current_subtask=current_subtask,
        task_description=task_description,
        available_objects=available_objects or [],
    )

    for attempt in range(1, max_retries + 1):
        _log(log_fn, f"[LLM Replan] Attempt {attempt}/{max_retries} with model={model}")
        try:
            response = call_llm_with_images(prompt, image_urls, model=model)
        except Exception as exc:  # noqa: BLE001
            _log(log_fn, f"[LLM Replan] API call failed: {exc}")
            continue
        if not response:
            _log(log_fn, "[LLM Replan] Empty response")
            continue

        plan = parse_recovery_plan(response, log_fn=log_fn)
        if not plan:
            _log(log_fn, "[LLM Replan] Parsed plan empty")
            continue
        _log(log_fn, f"[LLM Replan] Generated {len(plan)} recovery subtasks")
        return plan, True

    return [], False


def get_default_recovery_plan(current_subtask: Optional[Dict]) -> List[Dict]:
    """Fallback recovery plan when LLM replanning fails."""
    _ = current_subtask
    return [{"primitive": "init", "args": [], "instruction": "init()"}]

