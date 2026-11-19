#!/usr/bin/env python3
"""Generate synthetic spatial labels and box predicates for Rust DiskANN."""

import argparse
import json
from pathlib import Path

import numpy as np

from generate_metadata import (
    generate_clustered,
    generate_hollow,
    generate_normal,
    generate_skewed,
    generate_uniform,
)


DISTRIBUTIONS = ("uniform", "normal", "clustered", "skewed", "hollow")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def seed_value(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("must be between 0 and 4294967295")
    return number


def volume_fraction(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return number


def make_metadata(count, dimensions, distribution, seed):
    """Reuse CubeGraph's distributions, including its float32 serialization."""
    np.random.seed(seed)
    generators = {
        "uniform": lambda: generate_uniform(count, dimensions),
        "normal": lambda: generate_normal(count, dimensions),
        "clustered": lambda: generate_clustered(count, dimensions, seed=seed),
        "skewed": lambda: generate_skewed(count, dimensions, seed=seed),
        "hollow": lambda: generate_hollow(count, dimensions, seed=seed),
    }
    data = np.asarray(generators[distribution](), dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("metadata generator produced non-finite coordinates")
    return data


def make_boxes(metadata, count, volume, seed):
    """Yield contained boxes with the requested relative geometric volume."""
    lower = metadata.min(axis=0).astype(np.float64)
    upper = metadata.max(axis=0).astype(np.float64)
    width = (upper - lower) * volume ** (1.0 / metadata.shape[1])
    rng = np.random.default_rng(seed)
    for _ in range(count):
        if volume == 1:
            yield lower.astype(np.float32), upper.astype(np.float32)
        else:
            lo = lower + rng.random(metadata.shape[1]) * (upper - lower - width)
            yield lo.astype(np.float32), (lo + width).astype(np.float32)


def box_predicate(fields, lower, upper):
    terms = []
    for field, lo, hi in zip(fields, lower, upper):
        terms.append({field: {"$gte": float(lo)}})
        terms.append({field: {"$lte": float(hi)}})
    return {"$and": terms}


def write_json_line(stream, value):
    stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def generate(args):
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    data = make_metadata(args.num_base, args.dimensions, args.distribution, args.seed)
    fields = [f"a{axis}" for axis in range(args.dimensions)]
    output.mkdir(parents=True, exist_ok=False)

    with (output / "data_labels.jsonl").open("x", encoding="utf-8") as stream:
        for doc_id, row in enumerate(data):
            label = {"doc_id": doc_id}
            label.update(zip(fields, map(float, row)))
            write_json_line(stream, label)

    with (output / "query_predicates.jsonl").open("x", encoding="utf-8") as stream:
        boxes = make_boxes(data, args.num_queries, args.box_volume, args.seed)
        for query_id, (lower, upper) in enumerate(boxes):
            write_json_line(
                stream,
                {"query_id": query_id, "filter": box_predicate(fields, lower, upper)},
            )

    manifest = {
        "format": "diskann-spatial-jsonl-v1",
        "num_base": args.num_base,
        "num_queries": args.num_queries,
        "dimensions": args.dimensions,
        "distribution": args.distribution,
        "seed": args.seed,
        "box_volume": args.box_volume,
        "box_volume_semantics": "fraction of observed metadata bounding-box volume; not cardinality",
        "fields": fields,
        "coordinate_dtype": "float32",
        "domain_lower": data.min(axis=0).tolist(),
        "domain_upper": data.max(axis=0).tolist(),
        "data_labels": "data_labels.jsonl",
        "query_predicates": "query_predicates.jsonl",
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-base", type=positive_int, required=True)
    parser.add_argument("--num-queries", type=positive_int, required=True)
    parser.add_argument("--dimensions", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--distribution", choices=DISTRIBUTIONS, default="uniform")
    parser.add_argument("--box-volume", type=volume_fraction, default=0.01,
                        help="box volume / metadata domain volume, not selectivity (default: 0.01)")
    parser.add_argument("--seed", type=seed_value, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("diskann-data"),
                        help="new output directory (default: diskann-data)")
    args = parser.parse_args()
    try:
        generate(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Generated {args.num_base} labels and {args.num_queries} box predicates in {args.output_dir}")


if __name__ == "__main__":
    main()
