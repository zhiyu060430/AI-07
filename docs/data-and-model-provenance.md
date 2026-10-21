# Data and Model Provenance

The local runtime keeps large external assets under `/Volumes/T7/FirePhantom-runtime`; these files are not committed to Git.

| Asset | Source | License | Intended use |
| --- | --- | --- | --- |
| FASDD_CV v3, 95,126 images | [Roboflow project](https://universe.roboflow.com/forestfiresmoke/fasdd_cv-dx83j), mirrored at [Hugging Face](https://huggingface.co/datasets/seawsurf/fire_smoke_dataset_fasdd_cv) | CC BY 4.0, per the dataset card and packaged README | Fire/smoke detector training and held-out reference metrics; attribution is required |
| Tiny-SD | [segmind/tiny-sd](https://huggingface.co/segmind/tiny-sd) | CreativeML OpenRAIL-M | Local text-to-image synthesis; usage and distribution restrictions apply |
| YOLO11n pretrained checkpoint | [Ultralytics assets](https://github.com/ultralytics/assets/releases) | Ultralytics AGPL-3.0 terms; review the upstream license or obtain a commercial license before redistribution/deployment | Fire/smoke detector initialization and local MPS training |
| Blender 4.5.9 | [Blender downloads](https://www.blender.org/download/) | GNU GPL | Scene generation and rendering |
| Quaternius Ultimate Nature Pack, 150 models | [Quaternius](https://quaternius.com/packs/ultimatenature.html), file mirrored at [OpenGameArt](https://opengameart.org/content/low-poly-nature-pack-1) | CC0 | Default nature library (`assets/library/*.glb`). FBX converted locally to glb. Not committed to Git |
| Poly Haven props: Barrel_01, Barrel_02, barrel_03, wooden_crate_01, wooden_crate_02, small_lpg_tank, plastic_container, painted wooden sofa, WoodenTable_01, potted plant, boulder_01 | [Poly Haven](https://polyhaven.com/models) 1K glTF | CC0 | Room furniture and industrial props. Commercial use is allowed. Binaries stay under the assets root |
| Inception v3 weights | [PyTorch model weights](https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth) | BSD-style PyTorch/torchvision project terms | FID and IS feature/classifier backend |

Downloaded archive hashes and the source/attribution record are saved alongside the data under the T7 runtime. The reference set defaults to the FASDD_CV validation image split and uses a fixed, bounded sample. Do not use that same split as an independent accuracy test after tuning against its FID score.

The supplied FASDD_CV annotation export identifies two object classes, `fire` and `smoke`, and declares CC BY 4.0. Detector training writes a model and a held-out test-split mAP/precision/recall report. A short smoke run is not evidence for the SRS accuracy threshold; use a full training run and preserve the test report before making an acceptance claim.

The local YOLO11n checkpoint is stored outside the repository at `/Volumes/T7/FirePhantom-runtime/models/yolo11n.pt` (SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`). Its origin is the official Ultralytics pretrained YOLO11n asset. Keep the upstream license with any redistribution; a production or closed-source deployment may require Ultralytics' commercial license.

An earlier local DCGAN file may still sit at `/Volumes/T7/FirePhantom-runtime/models/flame_gan.pt`. The synthesis path does not load it.

An FDS 6.11.1 install may still sit under `/Volumes/T7/FirePhantom-runtime/tools/FDS`. The pipeline does not run it.
