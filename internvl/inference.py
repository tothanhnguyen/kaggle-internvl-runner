#!/usr/bin/env python3
"""Run zero-shot InternVL3.5 inference over a case-level TN-Mammo CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"A", "B", "C", "D"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError("PyYAML is required to read the experiment config; install it with `pip install pyyaml`") from error

    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")

    required = ("experiment", "dataset", "prompt", "generation", "models", "output", "evaluation")
    missing = [section for section in required if not isinstance(config.get(section), dict)]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    dataset = config["dataset"]
    views = dataset.get("view_columns")
    if views != ["left_cc", "left_mlo", "right_cc", "right_mlo"]:
        raise ValueError("View order must be left_cc, left_mlo, right_cc, right_mlo")

    prompt_path = resolve_path(config["prompt"].get("file", ""))
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Shared prompt file not found: {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Shared prompt file is empty: {prompt_path}")
    config["prompt"]["text"] = prompt_text

    models = config["models"]
    expected_models = {"qwen", "llava", "internvl"}
    if set(models) != expected_models:
        raise ValueError("models must define exactly qwen, llava, and internvl")
    for name, model_config in models.items():
        if not isinstance(model_config, dict) or not model_config.get("checkpoint"):
            raise ValueError(f"models.{name}.checkpoint is required")
        outputs = model_config.get("outputs")
        if not isinstance(outputs, dict) or not outputs.get("predictions_csv") or not outputs.get("raw_outputs_jsonl"):
            raise ValueError(f"models.{name}.outputs must define predictions_csv and raw_outputs_jsonl")

    internvl = models["internvl"]
    if internvl.get("quantization") != "none":
        raise ValueError("Only unquantized InternVL inference is currently implemented")
    if internvl.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("models.internvl.dtype must be bfloat16, float16, or float32")
    if internvl.get("device") not in {"auto", "cuda", "cpu"}:
        raise ValueError("models.internvl.device must be auto, cuda, or cpu")
    if not isinstance(internvl.get("image_size"), int) or internvl["image_size"] <= 0:
        raise ValueError("models.internvl.image_size must be a positive integer")
    if not isinstance(internvl.get("max_num_patches_per_view"), int) or internvl["max_num_patches_per_view"] < 1:
        raise ValueError("models.internvl.max_num_patches_per_view must be a positive integer")
    placeholders = internvl.get("image_placeholders")
    if not isinstance(placeholders, list) or len(placeholders) != 4 or any("<image>" not in item for item in placeholders):
        raise ValueError("models.internvl.image_placeholders must define four <image> placeholders")

    generation = config["generation"]
    if generation.get("do_sample") is not False:
        raise ValueError("The shared zero-shot protocol requires generation.do_sample=false")
    if not isinstance(generation.get("max_new_tokens"), int) or generation["max_new_tokens"] < 1:
        raise ValueError("generation.max_new_tokens must be a positive integer")

    metrics = config["evaluation"].get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("evaluation.metrics must be a non-empty list")
    return config


def read_cases(config: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    csv_path = resolve_path(dataset["csv"])
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    case_column = dataset.get("case_id_column", "case_id")
    label_column = dataset.get("label_column", "label")
    view_columns = dataset["view_columns"]
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {case_column, label_column, *view_columns}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset CSV is missing columns: {', '.join(sorted(missing))}")
        records = list(reader)

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        records = records[:limit]
    if not records:
        raise ValueError(f"Dataset contains no cases: {csv_path}")

    image_root_value = dataset.get("image_root")
    image_root = resolve_path(image_root_value) if image_root_value else csv_path.parent
    seen: set[str] = set()
    cases = []
    for row_number, row in enumerate(records, start=2):
        case_id = (row.get(case_column) or "").strip()
        if not case_id:
            raise ValueError(f"Blank case ID at CSV row {row_number}")
        if case_id in seen:
            raise ValueError(f"Duplicate case ID {case_id!r} in dataset CSV")
        seen.add(case_id)
        label = (row.get(label_column) or "").strip().upper()
        if label not in LABELS:
            raise ValueError(f"Invalid ground-truth label {label!r} at CSV row {row_number}")

        image_paths = []
        for column in view_columns:
            value = (row.get(column) or "").strip()
            if not value:
                raise ValueError(f"Missing {column} image for case {case_id!r}")
            path = resolve_path(value, image_root)
            if not path.is_file():
                raise FileNotFoundError(f"Image for case {case_id!r}, {column}: {path}")
            image_paths.append(path)
        cases.append({"case_id": case_id, "label": label, "image_paths": image_paths})
    return cases


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image: Any, image_size: int, max_num: int, use_thumbnail: bool) -> list[Any]:
    from PIL import Image

    width, height = image.size
    aspect_ratio = width / height
    target_ratios = {
        (columns, rows)
        for count in range(1, max_num + 1)
        for columns in range(1, count + 1)
        for rows in range(1, count + 1)
        if 1 <= columns * rows <= max_num
    }
    ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])
    columns, rows = _find_closest_aspect_ratio(aspect_ratio, ratios, width, height, image_size)
    target_width, target_height = image_size * columns, image_size * rows
    resized = image.resize((target_width, target_height))
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


def _load_image_tensor(path: Path, config: dict[str, Any], torch: Any) -> Any:
    import torchvision.transforms as transforms
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    model_config = config["models"]["internvl"]
    image_size = model_config["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    with Image.open(path) as image:
        image = image.convert("RGB")
        tiles = _dynamic_preprocess(
            image,
            image_size=image_size,
            max_num=model_config["max_num_patches_per_view"],
            use_thumbnail=model_config.get("use_thumbnail", False),
        )
        return torch.stack([transform(tile) for tile in tiles])


def _resolve_runtime(config: dict[str, Any], torch: Any) -> tuple[Any, Any]:
    configured_device = config["models"]["internvl"]["device"]
    device_name = configured_device
    if configured_device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected. Set model.device=cpu only if CPU inference is intentional "
                "and sufficient memory is available."
            )
        device_name = "cuda"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use an available CUDA host or set model.device=cpu")

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtypes[config["models"]["internvl"]["dtype"]]
    if device_name == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU inference is unsupported by this runner; use bfloat16 or float32")
    return torch.device(device_name), dtype


def _build_question(config: dict[str, Any]) -> str:
    placeholders = config["models"]["internvl"]["image_placeholders"]
    return "\n".join(placeholders) + "\n" + config["prompt"]["text"]


def _parse_label(response: str) -> str:
    candidate = response.strip().upper()
    return candidate if candidate in LABELS else "INVALID"


def run(config: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cases = read_cases(config, limit)
    model_config = config["models"]["internvl"]
    output_config = config["output"]
    predictions_path = resolve_path(model_config["outputs"]["predictions_csv"])
    raw_outputs_path = resolve_path(model_config["outputs"]["raw_outputs_jsonl"])
    if predictions_path == raw_outputs_path:
        raise ValueError("Prediction CSV and raw-output JSONL paths must differ")
    if not output_config.get("overwrite", False) and (predictions_path.exists() or raw_outputs_path.exists()):
        raise FileExistsError("Output exists; set output.overwrite=true or move existing output files")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    raw_outputs_path.parent.mkdir(parents=True, exist_ok=True)

    device, dtype = _resolve_runtime(config, torch)
    model = AutoModel.from_pretrained(
        model_config["checkpoint"],
        revision=model_config.get("revision", "main"),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=model_config.get("trust_remote_code", True),
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["checkpoint"],
        revision=model_config.get("revision", "main"),
        trust_remote_code=model_config.get("trust_remote_code", True),
        use_fast=False,
    )

    generation_config = {
        "do_sample": config["generation"]["do_sample"],
        "max_new_tokens": config["generation"]["max_new_tokens"],
        "num_beams": config["generation"].get("num_beams", 1),
    }
    question = _build_question(config)
    started = time.monotonic()
    valid_count = invalid_count = error_count = 0
    with predictions_path.open("w", newline="", encoding="utf-8") as predictions_stream, raw_outputs_path.open(
        "w", encoding="utf-8"
    ) as raw_stream:
        writer = csv.DictWriter(predictions_stream, fieldnames=["case_id", "pred_label"])
        writer.writeheader()
        for index, case in enumerate(cases, start=1):
            raw_response = None
            error_type = None
            prediction = "INVALID"
            pixel_values = None
            per_view = []
            try:
                per_view = [
                    _load_image_tensor(path, config, torch).to(device=device, dtype=dtype)
                    for path in case["image_paths"]
                ]
                num_patches_list = [tensor.size(0) for tensor in per_view]
                pixel_values = torch.cat(per_view, dim=0)
                with torch.inference_mode():
                    raw_response = model.chat(
                        tokenizer,
                        pixel_values,
                        question,
                        generation_config,
                        num_patches_list=num_patches_list,
                        history=None,
                    )
                if not isinstance(raw_response, str):
                    raise TypeError(f"Expected a text response, received {type(raw_response).__name__}")
                prediction = _parse_label(raw_response)
                if prediction == "INVALID":
                    invalid_count += 1
                else:
                    valid_count += 1
            except Exception as error:  # Keep an explicit row for every attempted case.
                error_type = type(error).__name__
                error_count += 1
            finally:
                pixel_values = None
                per_view.clear()

            writer.writerow({"case_id": case["case_id"], "pred_label": prediction})
            raw_stream.write(
                json.dumps(
                    {
                        "case_id": case["case_id"],
                        "raw_output": raw_response,
                        "pred_label": prediction,
                        "error_type": error_type,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            predictions_stream.flush()
            raw_stream.flush()
            print(f"[{index}/{len(cases)}] {case['case_id']}: {prediction}", file=sys.stderr)

    summary = {
        "checkpoint": model_config["checkpoint"],
        "revision": model_config.get("revision", "main"),
        "split": config["experiment"].get("split"),
        "cases": len(cases),
        "valid": valid_count,
        "invalid": invalid_count,
        "errors": error_count,
        "duration_seconds": round(time.monotonic() - started, 3),
        "predictions_csv": str(predictions_path),
        "raw_outputs_jsonl": str(raw_outputs_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml", help="Experiment YAML file")
    parser.add_argument("--limit", type=int, help="Run only the first N cases (pilot)")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate the YAML configuration without requiring dataset or model dependencies",
    )
    args = parser.parse_args()
    try:
        config_path = resolve_path(args.config)
        config = load_config(config_path)
        if args.check_config:
            print(f"Configuration valid: {config_path}")
            return 0
        summary = run(config, args.limit)
        return 1 if summary["errors"] else 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
