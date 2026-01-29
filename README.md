# Digital Signal Processing for Industrial Physics

This repository accompanies the **Digital Signal Processing for Industrial Physics** elective.  It provides a codes‑pace–ready set of Python demonstrations, datasets and completion checks for every week of the course.  The emphasis is on **black‑box operational understanding**—each notebook or script shows what goes into the DSP block, which knobs are tuned, what comes out, and how it can fail under industrial constraints.

The course covers sampling, filtering, spectral estimation, time–frequency methods, multichannel analysis, feature extraction and data‑driven approaches to pattern recognition and anomaly detection.  Each lesson folder contains a minimal demonstration (`demo.py`), a README explaining the objective, and a completion check script (`checks.py`) to validate that the learner has tuned parameters and measured metrics as required by the syllabus.

## Structure

- `dsp_utils/` – Shared helper functions for generating synthetic datasets, computing metrics and plotting.  Utilities in this package are used by the lesson demos to ensure consistent behaviour across weeks.
- `lessons/lesson_XX/` – Each weekly module has a corresponding lesson folder (e.g. `lesson_01`, `lesson_02`, … `lesson_14`).  Inside each folder you will find:
    - `README.md` – A brief summary of the week’s DSP topic and instructions for running the demonstration.
    - `demo.py` – A runnable script that synthesises data, applies the relevant DSP methods and prints or visualises the results.  These scripts are designed to work in a Codespaces environment with no internet access; all data is generated or cached locally.
    - `checks.py` – A small test module that verifies the learner’s work against the completion criteria defined in the syllabus.  Running `python checks.py` should return without assertion failures when the demo is complete.
- `slides_spec/` – A collection of JSON specifications that describe every slide in the course.  These specs are consumed by a separate slide‑rendering agent and are not actual presentations.  Each file follows the schema defined in the syllabus.
- `curriculum_manifest.json` – A manifest linking each week to its slide specification, lesson folder, dataset sources and completion checks.  This file is used by the QA agent to verify alignment between slides, code and datasets.
- `qa_report.json` – A summary of quality‑assurance checks across the entire curriculum.  It indicates whether each week’s slide spec, code and alignment tests passed the acceptance criteria.

## Getting Started

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Navigate to a lesson folder and run the demonstration.  For example, for Week 1:

   ```bash
   cd lessons/lesson_01
   python demo.py
   ```

   This will generate synthetic sensor signals, apply sampling and quantisation experiments and display summary metrics.  You can modify the tuning parameters at the top of each script to experiment with different settings.

3. Run the completion check to verify that your work meets the criteria:

   ```bash
   python checks.py
   ```

   If the script exits without errors, you have satisfied the week’s minimum completion requirements.

## License

This teaching material is released for educational purposes.  Please consult your instructor or the course syllabus for details on permitted use.