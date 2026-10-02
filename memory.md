# Project memory

- [verified] 2026-10-02: Forked StoneT2000/simple-easyhec into AprilRoboticsAI/simple-easyhec; SAM 3.1 work starts from upstream 69d1665bd20d86f986a477d8a0b5d7275dd0b752.
- SAM 3.1 backend, mask annotation CLI, regression tests, and isolated Python 3.12 environment moved from april_rl. Keep generic segmentation here; robot capture and calibration integration remain in april_rl.
- Preserve the pinned SAM source and checkpoint revisions and text-mask refinement behavior. No CUDA is available on this workstation for real-model validation.
- [verified] Six mocked SAM regression tests passed before and after relocation; real-model test skipped. Locked environment sync and standalone module help passed. april_rl launched this module successfully on an already-annotated/empty dataset without loading the GPU model. All dependency versions preserved.
