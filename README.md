# CubeGraph: A Multi-Resolution Graph Index for Approximate k-Nearest Neighbor Search with Geometric Filters

A high-performance C++ library for filtered approximate nearest neighbor (ANN) search using hierarchical cube-based spatial partitioning. Built on HNSWlib with SIMD acceleration (AVX2).

## Synthetic spatial workloads for Rust DiskANN

This fork adds a standalone generator; the original CubeGraph code is retained.
Only NumPy and Python 3 are needed, not the C++ build or any vector files:

```bash
python3 -m pip install -r scripts/requirements-diskann.txt
python3 scripts/generate_diskann.py --num-base 1000000 --num-queries 1000
```

The two counts are the only required arguments. The default output is a **new**
`diskann-data/` directory containing:

- `data_labels.jsonl`: one label per base vector, with zero-based `doc_id`.
- `query_predicates.jsonl`: one inclusive box predicate per query, with
  zero-based `query_id`.
- `manifest.json`: generation parameters, attribute names, and observed domain.

For example, the JSONL records have these shapes:

```json
{"doc_id":0,"a0":0.42,"a1":0.67}
{"query_id":0,"filter":{"$and":[{"a0":{"$gte":0.2}},{"a0":{"$lte":0.8}},{"a1":{"$gte":0.3}},{"a1":{"$lte":0.7}}]}}
```

Defaults are uniform 2D coordinates, box-volume fraction 0.01, and seed 42.
All five original distributions and 2D/3D/4D attributes are supported:

```bash
python3 scripts/generate_diskann.py \
  --num-base 1000000 --num-queries 1000 \
  --dimensions 3 --distribution clustered --box-volume 0.05 \
  --seed 123 --output-dir workloads/clustered-3d
```

`--distribution` accepts `uniform`, `normal`, `clustered`, `skewed`, or `hollow`.
Coordinates reuse CubeGraph's generators and are rounded to float32 before
JSON serialization. Normal/clustered coordinates are clipped; skewed uses
Beta(0.5, 0.5); hollow uses random directions and radii in [0.2, 0.5].
Each box has equal fractional side lengths relative to the **observed metadata
bounding box**, with a random contained position. Bounds are float32 and inclusive.
Unlike the original perturbed benchmark, no +/-20% side variation is applied.
`--box-volume` must be in (0, 1]; it is a geometric-volume fraction, **not an
exact qualifying-point fraction**, especially for nonuniform distributions.
Float32 rounding may change the volume slightly. Constant dimensions produce
degenerate intervals; with one base point, every box includes that point.
Generation is reproducible with the same arguments and NumPy version, but is
not intended to reproduce the C++ random-number sequence.

Use the files as `filter_params.data_labels` and
`filter_params.query_predicates` in the Rust DiskANN benchmark, with a
range-capable bitmap filter path. The attribute-ID-encoded inline-beta path
does not support `$gte`/`$lte`. These are **not** legacy C++ categorical-label
files. Vector row `i` must correspond to `doc_id: i`, and query vector row `j`
to `query_id: j`; no dataset-specific names or embeddings are required.

This tool generates spatial metadata and filters only: it does not generate
embeddings, query vectors, eligibility bitmaps, or nearest-neighbor ground truth.
Those require the corresponding vector data and/or DiskANN's filtering tools.
Metadata occupies O(base count * attribute dimension) memory; JSONL records
are written incrementally. Existing output directories are never overwritten.
`manifest.json` is written last; a failed write may leave an incomplete output
directory, which must be removed or replaced with a fresh destination before retrying.
Generated JSONL is larger than CubeGraph's binary metadata.

Run the generator tests without building CubeGraph:

```bash
python3 -m unittest discover -s tests -p 'test_generate_diskann.py' -v
```

## Paper and reproducibility

- [Paper source and submission PDF](https://github.com/mingyu-hkustgz/Cube-Graph-paper)
- [Technical report: complete paper, references, and Appendices A/B](technical_report.pdf)
- [Shared box, radius, layer and update experiments](experiments/README.md)
- [Audited predetermined/on-the-fly comparison](experiments/search_modes/README.md)
- [Audited high-aspect-ratio results and protocol](experiments/aspect_ratio/README.md)
- [Complete 0.5% extension, validity checks and results](experiments/selectivity/README.md)

The reported benefits depend on the workload. At 0.5% qualifying fraction,
CubeGraph beats KHI in nine of ten box configurations, including all six
3D/4D configurations (1.18–3.79×); YFCC 2D favors KHI (CG/KHI = 0.54×).
Only CubeGraph reaches 99% recall in every repeat on the four tested 0.5%
ball/exclusion workloads. Predicate-invalid ACORN measurements are retained
as failures and excluded from valid performance curves. At 1% qualifying fraction,
CubeGraph is 1.41–3.89× faster than KHI on all six measured 3D/4D box
configurations. For SIFT 2D circle and box-minus-circle filters at 2%, it is
3.25×/3.16× faster than ACORN and 5.45×/5.59× faster than POST at 99% recall
in every repeat. The complete report retains broader-filter reversals and
configurations favoring KHI. Complex-predicate comparisons do not include KHI.
On MSMARCO10M 4D at 1%, the 2.17x query-throughput gain over KHI accompanies
53.9% less serialized index space (8.966 versus 19.467 GB). The complete
high-aspect-ratio sweep covers SIFT1M and YFCC, ratios 1/4/16/32 and both modes;
all 16 combinations reach 99% recall, with lower throughput on thinner boxes.


## Key Features

- **Hierarchical Cube Partitioning**: Multi-level spatial indexing that adapts to filter size
- **Cross-Cube Edges**: HNSW extended with edges to adjacent cubes for seamless traversal
- **Multiple Filter Types**: BBox (axis-aligned bounding box), Radius (sphere), and Polygon filters
- **Composite Filters**: AND, OR, NOT compositions for complex filter predicates
- **Two Search Strategies**:
  - `fly_search`: Dynamic cube traversal starting from filter center
  - `predetermined_search`: Precomputed cube list based on BFS from filter region
- **SIMD Acceleration**: AVX2-optimized distance calculations
- **Adaptive Layer Selection**: Automatically selects appropriate cube resolution based on filter size

## Architecture

### Core Components

1. **IndexCube** (`include/IndexCube.h`): Main index with hierarchical cube partitioning
2. **HierarchicalNSWCube** (`third_party/hnsw-cube.h`): Extended HNSW with cross-cube edges
3. **Filter Functors**: BBoxFilter, RadiusFilter, PolygonFilter (implement `BaseFilterFunctor`)
4. **Composite Filters**: AndFilter, OrFilter, NotFilter for filter composition

### Data Formats

- **Vector files**: fvecs format (`[dim: int32][vec[0]: float]...`)
- **Metadata files**: Binary format (`[n: size_t][d: size_t][vector_0: float[d]]...`)

### Hierarchical Cube Structure

Each layer divides attribute space into `(2^(layer_id+1))^attr_dim` cubes:
- Layer 0: 2 cubes per dimension (4 total for 2D)
- Layer 1: 4 cubes per dimension (16 total for 2D)
- Layer 2: 8 cubes per dimension (64 total for 2D)

Box selection compares each box side with the corresponding cube width;
radius and polygon selectors use their implemented diagonal-based rules:
- Large filters → coarser layers (fewer, larger cubes)
- Small filters → finer layers (more, smaller cubes)

## Bottom-up construction

The finest adaptive layer is built first. Each parent layer preserves child
internal IDs and label mappings, inherits local neighbors, and promotes child
cross edges whose endpoints belong to the same parent cube. Child entry points
provide candidate connections between siblings, including when cross edges
are disabled for ablations.

Each parent performs one synchronous refinement round over an immutable
inherited graph with beam width `min(ef_construction, 4*M)` (64 at the default parameters), followed by reciprocal-edge
repair and degree pruning. Cross edges are then built between adjacent parent
cubes. Serialization and query layer order remain coarse to fine. Old index
files remain readable; rebuilding is required to obtain the new graphs.

Run `bash tests/run_bottom_up.sh` on an x86 Linux host with GCC and OpenMP.
The test covers 2D/3D/4D, cross edges on/off, constant attributes, neighbor
ownership and degree bounds, label mappings, save/load equality, and exact
synthetic recall at exhaustive search width. The vendored HNSWlib headers and
license are included so a fresh clone has the complete dependency.

## Building

### Requirements

- C++17 compiler with AVX2 support
- CMake 3.15+
- Boost (program_options, filesystem)
- OpenMP

### Build Instructions

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Basic Example

```cpp
#include "IndexCube.h"

using namespace hnswlib;

// Create index
IndexCube index(/*num_layers=*/4, /*M=*/16, /*ef_construction=*/200, /*cross_edge_count=*/2);

// Build index from data and metadata files
index.build_index("data.fvecs", "metadata.bin");

// Define bounding box filter
BoundingBox bbox(2);
bbox.min_bounds = {0.2f, 0.3f};
bbox.max_bounds = {0.8f, 0.7f};

// Search with fly_search (dynamic traversal)
auto results = index.fly_search(query_vector, k, &bbox);

// Or use predetermined_search (precomputed cube list)
auto results = index.predetermined_search(query_vector, k, &bbox);
```

### Filter Types

#### BBox Filter (Axis-Aligned Bounding Box)

```cpp
BoundingBox bbox(attr_dim);
bbox.min_bounds = {x_min, y_min, ...};
bbox.max_bounds = {x_max, y_max, ...};

auto results = index.fly_search(query_vector, k, &bbox);
```

#### Radius Filter (Sphere)

```cpp
RadiusFilterParams radius;
radius.sphere.center = {0.5f, 0.5f};
radius.sphere.radius = 0.2f;
radius.attr_dim = 2;

auto results = index.fly_search(query_vector, k, &radius);
```

#### Polygon Filter (2D only)

```cpp
PolygonFilterParams polygon;
polygon.vertices = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, 1.0f}};
polygon.num_vertices = 3;
polygon.attr_dim = 2;

auto results = index.fly_search(query_vector, k, &polygon);
```

#### Composite Filters (AND/OR/NOT)

```cpp
CompositeFilterParams params;
params.bbox = bbox;
params.spec.use_bbox = true;
params.attr_dim = 2;

RadiusFilterParams radius;
radius.sphere = Sphere({0.5f, 0.5f}, 0.1f);
radius.attr_dim = 2;

params.radius = radius;
params.spec.use_radius = true;

// Default: BBox AND Radius
auto results = index.fly_search(query_vector, k, &params);
```

### Python Scripts

```bash
cd scripts
pip install -r requirements.txt

# Generate test data
python generate_metadata.py --type uniform --count 10000 --attr-dim 2

# Visualize metadata distribution
python visualize_metadata.py <metadata.bin>
```

## Project Structure

```
CubeGraph/
├── CMakeLists.txt
├── README.md
├── include/
│   └── IndexCube.h              # Main index with hierarchical cubes
├── src/
│   ├── bench_hierarchical_cube.cpp    # BBox filter benchmarks
│   ├── bench_hierarchical_ball.cpp      # Radius filter benchmarks
│   ├── bench_hierarchical_polygon.cpp   # Polygon filter benchmarks
│   ├── bench_post_filtering.cpp        # Post-filtering comparison
│   ├── bench_complex_filter.cpp        # Composite filter benchmarks
│   └── ...
├── third_party/
│   ├── hnswlib/                 # HNSWlib base
│   ├── hnsw-cube.h              # Extended HNSW with cross-cube edges
│   ├── matrix.h                 # Matrix utilities
│   └── ...
└── scripts/
    ├── generate_metadata.py     # Generate test metadata
    ├── visualize_metadata.py    # Visualize metadata distribution
    └── ...
```

## Search Algorithms

### Fly Search

Dynamic traversal starting from the cube containing the filter center:
1. Select layer based on filter size vs cube diagonal
2. Start from the cube containing filter center (or centroid for polygon)
3. Traverse HNSW within current cube
4. Follow cross-cube edges to adjacent cubes only if they contain in-filter nodes
5. Continue until all accessible cubes are explored

### Predetermined Search

Precomputed cube list based on BFS from filter region:
1. Select layer based on filter bounding box
2. BFS from central cube, adding all cubes that overlap filter region
3. Search HNSW across all precomputed cubes in parallel

## Build Flags

The project uses aggressive optimization flags:
- `-Ofast -march=core-avx2 -mavx512f -fopenmp -ftree-vectorize`

These require a CPU with AVX2 support.

## References

**CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data** [[paper](https://arxiv.org/abs/2604.06616)] [[bibtex](#bibtex-cite)] - Yang et al., 2026

If you use **CubeGraph** in a research paper, please cite:

```
@article{yang2026cubegraph,
  title={CubeGraph: Efficient Retrieval-Augmented Generation for Spatial and Temporal Data},
  author={Mingyu Yang and Wentao Li and Wei Wang},
  year={2026},
  eprint={2604.06616},
  archivePrefix={arXiv},
  primaryClass={cs.DB},
  url={https://arxiv.org/abs/2604.06616}
}
```

<span id="bibtex-cite"></span>
