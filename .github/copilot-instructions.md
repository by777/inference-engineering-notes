# Repository Guidelines

## Build and run

- There is no root-level aggregate build, test, or lint command. Lessons are independent experiments; read that lesson's `SUMMARY.md`, `STEPS.md`, or `README.md` first, then use its `Makefile` or documented script.
- Build and run one Makefile-based lesson from the repository root with `make -C "lesson-01-ORT-最小自定义算子" run`. Most Makefiles also expose `clean`; check the lesson's Makefile for its actual targets.
- Run one focused ONNX Runtime integration check with `make -C "lesson-02-ORT-打包成动态库给python加载" run`. To rerun just its Python check after generating the model and shared library, use `python3 "lesson-02-ORT-打包成动态库给python加载/test_model.py"`. Lessons 02–06 use similar standalone `test_model.py` scripts; these are not pytest tests.
- Other Python experiments are run individually with `python3 <script.py>` from the lesson directory; use the lesson's own instructions for any generated inputs or required runtime.
- ONNX Runtime examples need the matching headers and library under `ort-bin/` (or the documented `ORT_DIR` setup). TVM, MNN, and MLIR source/build trees are local prerequisites, not distributed with the repository. ARM cross-compilation also requires an Android NDK configured through `NDK`.

## Architecture

- The root README organizes a progression from runtime behavior to compiler and deployment: ONNX Runtime custom operators, sessions, profiling, and I/O binding in lessons 01–12; TVM lowering, tuning, and deployment in 13–18; MNN and ARM/NEON experiments in 19–22.
- Lesson 23 explores MLIR dialects and passes. Lessons 21–22 connect quantization fundamentals to MNN implementation; lessons 24–25 cover inference engineering, performance modeling, and numeric formats.
- Lessons are intentionally runnable and readable in isolation. Generated models, binaries, and experiment outputs are local artifacts, not shared build products; use the lesson's generation and cleanup targets where provided.
- `learn-inference/` is an offline reading copy of third-party material. Keep repository-authored explanations in the lesson notes, respect the original rights, and do not add or reproduce substantial third-party text.

## Repository conventions

- Prefer the smallest experiment that demonstrates a claim: keep code, model-generation steps, and the explanation together in the relevant lesson rather than introducing a repository-wide framework.
- Scripts should derive paths from the repository/lesson location or documented environment variables, not from assumptions about a particular clone location. The documented shared variables include `REPO`, `ORT_DIR`, and `NDK`; local toolchains may live under `$HOME/toolchain/`.
- Keep technical notes evidence-based: make conclusions reproducible where possible, label uncertain claims as hypotheses or pending verification, and cite public sources for product performance figures.
- Check `.gitignore` before adding outputs. Model files, shared libraries, executables, local dependency trees, caches, and experiment logs are generally generated or machine-specific and should not be committed.
