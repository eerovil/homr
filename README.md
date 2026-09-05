# homr

homr is an Optical Music Recognition (OMR) software designed to transform camera pictures of sheet music into
machine-readable MusicXML format. The resulting [MusicXML](https://www.w3.org/2021/06/musicxml40/) files can be further
processed using tools such as [musescore](https://musescore.com/).

For a quick try, visit our online demo at [homr.site](https://homr.site) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liebharc/homr/blob/main/colab.ipynb)

You might also want to check out [Andromr](https://github.com/aicelen/Andromr), an Android app for optical music recognition using homr.

## About this fork

`eerovil/homr` is a fork of [liebharc/homr](https://github.com/liebharc/homr), kept
permanently rather than as a staging area. It exists to read Finnish choral scans for
[musescore-choir-plugins](https://github.com/eerovil/musescore-choir-plugins), which
installs this fork's `main` and calls it as a subprocess.

**Where a fix belongs.** A defect in a scanned score can be repaired here or in the
choir app, and this is the rule for deciding (settled on
[musescore-choir-plugins#141](https://github.com/eerovil/musescore-choir-plugins/issues/141)):

> **homr's job is to produce MusicXML that, when rendered, looks like the original
> page** — including stem direction, which is to say the voices. Anything after that
> point belongs to the choir app.

So if homr got the page wrong, the fix is here, whether or not the evidence is still in
the pixels — a slur nobody engraved, two noteheads read as one, staves grouped into the
wrong system. If the parse already matches the page and the app wants something further
from it — which band to read next, what the operator is shown, how a practice track is
built — that is the app's, not homr's. The rule is a claim about the *output*, so it is
testable: render the MusicXML and hold it against the page.

The choir app currently has some OMR repair code of its own that predates this rule and
is on the wrong side of it. That is being migrated; the rule binds new fixes.

**Upstreaming is opportunistic.** Fixes here are general improvements often enough, and
one is welcome to go to `liebharc/homr` — but nothing is tracked, nothing is owed, and
no decision here waits on upstream review. Note that `CONTRIBUTING.md` is upstream's
file and describes contributing to *upstream*, not to this fork.

**The measurement harness lives here** (`fixturecheck/`, `scripts/choir-bench.py`,
`scripts/choir-worktree.sh`, `scripts/choir-k8s.sh`). It has to run inside homr's own
environment, and it reaches into the choir repo through `CHOIR_REPO` for the fixtures
and reference scores it judges against. See `scripts/README.md`.

## Prerequisites

- Python 3.11 or 3.12
- Poetry or UV
- Optional
  - NVIDIA GPU with CUDA 12.1
  - AMD GPU with ROCm 7.0

## Getting started (uv)

The easiest way to get started is using `uvx` (`uv` must be installed). Select an inference backend:
- CPU: `uvx --from 'homr[cpu]' homr <image>`
- NVIDIA CUDA: `uvx --from 'homr[cuda]' homr <image>`
- AMD ROCm: `uvx --python 3.12 --from 'homr[rocm]' homr <image>`

Then see the resulting MusicXML:
- It will be saved in the same directory as the input image
- To combine the MusicXML results from multiple images, you can use [relieur](https://github.com/papoteur-mga/relieur)

## Getting started (poetry)

- Clone the repository
- Install dependencies for:
  - Inference: `poetry install --only main --extras cpu`
  - Development: `poetry install --extras cpu`
  - If using GPU, replace `--extras cpu` to `--extras cuda` / `--extras rocm`
- Run the program using `poetry run homr <image>`
- The resulting MusicXML file will be saved in the same directory as the input image
- To combine the MusicXML results from multiple images, you can use [relieur](https://github.com/papoteur-mga/relieur)

### Optional score settings

When you know notation facts about one score, keep them in a JSON file and opt in for that run:

```json
{
  "decoder": {"minimum_duration": "16", "allow_tuplets": true, "allow_grace_notes": true},
  "postprocessing": {"stem_voice_hints": true}
}
```

```bash
poetry run homr page.png --score-settings page.score-settings.json --output-confidence
```

`minimum_duration` is the shortest written base note value, so `16` permits dotted
16ths and 16th-note tuplets but excludes 32nds. `--output-confidence` writes a
`.confidence.json` sidecar with the selected decoder tokens, their probabilities,
and their nearest alternatives. When a constraint changes a rhythm choice, that
record also retains the unconstrained result.

`stem_voice_hints` is **on** in this fork; set it to `false` to turn it off for a
score. Each decoded note is matched to the notehead segmentation found for it,
and where a staff is carrying two voices in a bar, a note drawn with an up stem
becomes voice 1 (or 5 on staff 2) and a down stem voice 2 (or 6).

A staff is taken to be carrying two voices when two notes sound at one moment
with their stems drawn opposite ways, when a stem contradicts the note's height
-- an up stem above the middle line, or a down stem below it -- or when one
printed notehead carries both stems. Anywhere else, and wherever the match to a
notehead is not clean, the existing voice assignment stands: on a staff carrying
one voice the stems say how high the notes are, not which voice they are, and
following them there would split that voice in half.

## Example

The example below provides an overview of the current performance of the implementation. While some errors are present
in the output, the overall structure remains accurate.

|                                          Original Image                                           |                                                                               homr Result                                                                                |
| :-----------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/BreezeWhite/oemer/blob/main/figures/tabi.jpg?raw=true" width="400" > | <img src="https://github.com/liebharc/homr/blob/main/figures/tabi.svg?raw=true" alt="Go to https://github.com/liebharc/homr if this image isn't displayed" width="400" > |

The homr result is obtained by processing the [homr output](figures/tabi.musicxml) and rendering it
with [musescore](https://musescore.com/).

## Limitations

The current implementation focuses on pitch and rhythm information on the bass or treble clef, neglecting dynamics,
articulation, double sharps/flats, and other musical symbols.

## Technical Details

homr uses a two-stage pipeline: **segmentation** for structural analysis followed by **semantic symbol recognition** via transformer models.

### Stage 1: Image Segmentation and Structural Analysis

homr employs UNet-based segmentation models (adapted from [oemer](https://github.com/BreezeWhite/oemer)) to extract structural components from the sheet music image:

- **Staff lines and symbols**: Detected via trained segmentation networks that identify:
  - Staff line fragments
  - Note heads
  - Stems and rests
  - Bar lines
  - Clefs and key signatures

The segmentation process generates bounding boxes for each detected element. These predictions serve as inputs for the staff detection algorithm.

### Stage 2: Staff Detection and Merging

Using the segmentation outputs, homr constructs staffs through the following steps:

1. **Staff Anchor Detection**: The algorithm identifies "staff anchors" (clefs and bar lines) that serve as reference points for accurate staff localization, even when symbols partially obscure staff lines.

2. **Unit Size Estimation**: For each staff, the algorithm calculates the "unit size" (distance between staff lines). This accommodates camera perspective variations and non-uniform staff spacing.

3. **Staff Reconstruction**: Around each anchor, five staff lines are located and the remaining staff structure is reconstructed using the estimated unit size.

4. **Grand Staff Merging**: Braces and brackets are identified to merge related staffs, supporting:
   - Grand staffs (piano, organ)
   - Multiple voices on a single staff
   - Mixed instrument groups

### Stage 3: Semantic Symbol Recognition via Transformer

Each staff is dewarped (perspective-corrected) and passed through a transformer-based model (based on [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR)) that performs **end-to-end symbol sequence recognition**. The model outputs:

- **Rhythm symbols**: Note durations, rests, and tuplet information
- **Pitch information**: Absolute pitch values with accidentals (sharps, flats, naturals)
- **Articulation marks**: Accents, staccato, tenuto, and slur markers
- **Performance annotations**: Dynamic expressions and other musical notation

The transformer model generates these predictions in sequence, processing the dewarped staff image to understand the spatial and temporal relationships between musical symbols.

**Note**: The transformer output provides the sequence of symbols but does not include explicit positional information (horizontal or vertical coordinates). However, the model computes the center of attention as a byproduct of the attention mechanism, which can be used to estimate the focus point on the staff image.

### Stage 4: MusicXML Output

The symbol sequence is converted into MusicXML format and saved to disk. The resulting file can be processed with tools like [musescore](https://musescore.com/) or [relieur](https://github.com/papoteur-mga/relieur) (for multi-image combinations).

## Citation

If you use this code in your research work, please cite [oemer](https://github.com/BreezeWhite/oemer)
and [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR).

## Name

The name "homr" stands for Homer's Optical Music Recognition (OMR), leaving the interpretation of "Homer" to the user's
discretion, whether referring to the ancient poet [Homer](https://en.wikipedia.org/wiki/Homer) or the iconic character
from [The Simpsons](https://en.wikipedia.org/wiki/The_Simpsons).

## Thanks

This project builds upon previous work, including:

- The segmentation models of [oemer](https://github.com/BreezeWhite/oemer)
- The transformer model of [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR)
- The starter template provided by [Benjamin Roland](https://github.com/Parici75/python-poetry-bootstrap)
