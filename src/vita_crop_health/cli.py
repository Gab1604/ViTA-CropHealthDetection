"""Command-line interface for the ViTA crop-health MVP."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Sequence

from .analyzer import AnalysisConfig, analyze_crop_health
from .io import load_array, save_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vita-crop-health",
        description=(
            "Calculate NDVI, GNDVI, SAVI, EVI, NDWI and a crop vigor/anomaly score "
            "from co-registered Blue, Green, Red and NIR arrays."
        ),
    )
    parser.add_argument("--red", required=True, help="Current Red-band file.")
    parser.add_argument("--green", required=True, help="Current Green-band file.")
    parser.add_argument("--blue", required=True, help="Current Blue-band file.")
    parser.add_argument("--nir", required=True, help="Current NIR-band file.")
    parser.add_argument("--crop-mask", required=True, help="Non-zero pixels are treated as crop.")
    parser.add_argument(
        "--cloud-mask",
        help="Optional mask where non-zero pixels mean cloud or cloud shadow and are excluded.",
    )

    parser.add_argument("--previous-red", help="Previous co-registered Red band.")
    parser.add_argument("--previous-green", help="Previous co-registered Green band.")
    parser.add_argument("--previous-blue", help="Previous co-registered Blue band.")
    parser.add_argument("--previous-nir", help="Previous co-registered NIR band.")
    parser.add_argument("--previous-cloud-mask", help="Optional previous cloud/cloud-shadow mask.")

    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "Reflectance scale denominator. Use 1 for reflectance in [0,1], 10000 for common "
            "Sentinel-2 L2A values, or 255 for normalized 8-bit data."
        ),
    )
    parser.add_argument(
        "--config",
        help="Optional JSON file overriding AnalysisConfig fields.",
    )
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument(
        "--no-quicklook",
        action="store_true",
        help="Do not attempt to create a PNG heatmap.",
    )
    return parser


def _load_config(path: str | None, scale: float | None) -> AnalysisConfig:
    values: dict[str, object] = {}
    if path:
        config_path = Path(path)
        values = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("The config file must contain a JSON object.")
        allowed = {item.name for item in fields(AnalysisConfig)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown config fields: {', '.join(unknown)}")
    if scale is not None:
        values["reflectance_scale"] = scale
    return AnalysisConfig(**values)


def _load_previous(args: argparse.Namespace) -> dict[str, object] | None:
    paths = {
        "red": args.previous_red,
        "green": args.previous_green,
        "blue": args.previous_blue,
        "nir": args.previous_nir,
    }
    supplied = [value is not None for value in paths.values()]
    if any(supplied) and not all(supplied):
        missing = [name for name, value in paths.items() if value is None]
        raise ValueError(
            "Temporal analysis requires all four previous bands. Missing: " + ", ".join(missing)
        )
    if not any(supplied):
        return None
    return {name: load_array(value) for name, value in paths.items() if value is not None}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = _load_config(args.config, args.scale)
        current_bands = {
            "red": load_array(args.red),
            "green": load_array(args.green),
            "blue": load_array(args.blue),
            "nir": load_array(args.nir),
        }
        crop_mask = load_array(args.crop_mask)
        cloud_mask = load_array(args.cloud_mask) if args.cloud_mask else None
        previous_bands = _load_previous(args)
        previous_cloud_mask = (
            load_array(args.previous_cloud_mask) if args.previous_cloud_mask else None
        )

        result = analyze_crop_health(
            bands=current_bands,
            crop_mask=crop_mask,
            cloud_mask=cloud_mask,
            previous_bands=previous_bands,
            previous_cloud_mask=previous_cloud_mask,
            config=config,
        )
        report_path = save_result(
            args.output,
            result,
            config=config,
            create_quicklook=not args.no_quicklook,
        )
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(json.dumps(result.to_dict(config), indent=2, allow_nan=False))
    print(f"\nSaved report to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
