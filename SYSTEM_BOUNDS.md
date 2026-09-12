# Printed-system bounds proposals

This fork can locate printed systems in a PDF without decoding their music. The
result is a proposal for an editor: homr does not save or approve the choir app's
`.systems.json`, write MusicXML, or decide which singer owns a detected staff.

## Command line

Run this fork from its Poetry environment, with Poppler's `pdftoppm` available on
`PATH`:

```sh
poetry run homr score.pdf --gpu no --find-system-bounds
poetry run homr score.pdf --gpu no --find-system-bounds --system-page 2
```

The default proposal raster resolution is **200 dpi**. `--system-dpi 200` makes
that choice explicit. Poppler is intentional: it preserves the rasterization
used to measure the choir grouping rule. PDFium is used to inspect page count,
not as an interchangeable rasterizer for this mode.

The CLI initializes the segmentation model when needed. Proposal mode does not
require or download the transformer encoder/decoder weights. Do not use `--init`
for a proposal-only invocation: `--init` is the separate full-model initialization
command.

On success, stdout contains one JSON object and a trailing newline; progress and
diagnostics go to stderr. Consume stdout only after checking the process exit
status. Invalid proposal arguments and PDF/rendering errors exit nonzero rather
than returning a successful empty proposal. `--system-page` is one-based and is
valid only with `--find-system-bounds`. Proposal mode cannot be combined with
`--system-bounds`, which reads already selected bands and decodes their music.

## JSON contract

Illustrative output for one system:

```json
{"systems": [{"index": 1, "page": 1, "top": 0.1, "bottom": 0.9, "measure_start": 0, "measure_end": 0}]}
```

`page` is a one-based PDF page number. `top` and `bottom` are fractions of the
original page height, not pixels. Systems are ordered by PDF page and then from
top to bottom. A whole-PDF request numbers `index` from one across the score.
A single-page request retains its actual PDF page number but numbers its systems
from one; a caller requesting pages independently must reindex the accumulated
results across the score. Measure numbers are not inferred by this mode and
remain zero.

Adjacent staves are grouped by agreement of their interior barlines. The measured
constants are `TOL_X=0.008`, `EDGE_X=0.02`, and `AGREE=0.6`; page gaps provide the
existing veto/fallback when barline evidence is incomplete. Adjacent proposal
bands meet halfway between systems, with room added at the outer page edges.
The operator must still inspect and correct proposals before saving them.

## Python API

Inside an environment with the segmentation model already initialized:

```python
from homr.system_finder import bounds_payload, find_system_bounds

bounds = find_system_bounds("score.pdf", use_gpu=False, dpi=200)
payload = bounds_payload(bounds)
```

`find_system_bounds(..., page=2)` applies the same single-page indexing rule as
the CLI. It returns `SystemBounds` values; `bounds_payload` supplies the JSON
schema shown above. Neither function saves or approves an application bounds
file. `detect_page_geometry`, `group_staves`, and `bands_for_page` expose the
segmentation and measured grouping stages without transformer decoding.

## Compatibility and verification

The choir application keeps its old helper as a compatibility fallback only
when the installed homr explicitly rejects `--find-system-bounds`. A supported
CLI failing to propose a page is an error, not permission to silently switch
implementations. The fallback stays until this fork has been merged, manually
installed, and re-measured; implementing this feature does not install or deploy
anything on the user's host.

Focused tests cover grouping, band edges, page order, JSON output, and
segmentation-only startup. The choir benchmark checks the unchanged hand-drawn
Virta fixture (15 systems) and B1a/B1b (3 systems each), with a 0.02 page-height
internal-boundary tolerance and outer bands covering the hand-drawn systems.
Canonical CI and all five real recognition fixtures must also pass without
changing their references.
