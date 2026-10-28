DINOv3 Adaptive Zoom: Uncertainty-Guided Semantic Segmentation
A graduation project exploring whether uncertainty-guided local reprocessing can improve semantic segmentation while using fewer model passes than dense sliding-window inference.

A graduation project that extends DINOv3-based semantic segmentation by automatically identifying uncertain regions in the global prediction and re-processing them at higher resolution (adaptive zoom). Experiments were conducted on Google Colab with a Tesla T4 GPU using the VOC2012 dataset.

## Method

1. Process the image at 224×224 to obtain global logits.
2. Compute a normalized entropy map from softmax probabilities using `log(21)` as the normalization factor.
3. Rank candidate crops by their mean entropy score and select the Top-K after applying an IoU filter.
4. Crop the selected regions from the **original RGB image**, resize them to the model's input size, and re-process them with the same model.
5. Map local logits back to the original coordinate space. Outside crops, retain global logits; inside crops, replace with local logits; in overlapping regions, average the local logits.

The selector does not use ground truth during inference. The model uses a frozen `facebook/dinov3-vits16-pretrain-lvd1689m` backbone and a 21-class 1×1 convolution head trained for 5 epochs on the VOC2012 training set. No retraining occurs during the demo.

Default configuration: **K=3, crop=112, stride=28, box IoU threshold=0.30, padding=0**. The crop size is the window size on the 224×224 selection map. The actual number of selected crops may be fewer than K after IoU filtering.

## Results

| Method | Pooled mIoU | Mean time (ms/image) | Backbone passes |
|---|---:|---:|---:|
| Baseline | 0.7133 | 79.2 | 1 |
| Adaptive K=1 | 0.7124 | 110.8 | 2 |
| Adaptive K=3 | 0.7220 | 147.5 | 4 |
| Random K=3, mean of three seeds | 0.7140 | 147.8 | 4 per seed |
| Sliding-window | 0.7325 | 510.3 | 25 |

**mIoU is computed over 1,399 images** at original resolution, including background, with ignore=255. **Timing is measured on 100 images** from the same session. Those 100 images are a subset of the main evaluation set. The random mean is averaged across seeds 42/43/44 — not an ensemble of predictions.

- K=3 − random K=3: **+0.80 percentage points**, 95% paired bootstrap interval **[+0.29; +1.33]**.
- K=3 − baseline: **+0.87 pp**, interval **[−0.14; +1.96]**. Superiority over baseline is not confirmed.
- K=3 − sliding: **−1.05 pp**, interval **[−1.51; −0.58]**. Sliding-window achieves higher accuracy.
- Timing benchmark sliding/K=3 ratio: **3.46×**, interval **[3.33; 3.62]**. The sliding-window implementation is sequential and unoptimized.

50 development-set images were used for configuration selection and excluded from the main evaluation. The main evaluation set is previously inspected validation data — it is **not** a held-out test set. Equal crop counts do not mean that adaptive and random methods cover the same image regions. Local re-processing can degrade some images.

Full protocol, ablations, per-class results, and limitations: [final experiment report](Docs/DINOv3_Final_Deney_Raporu_TR.md).

### Quality–cost trade-off and final selection

![Quality–cost on the same 100 images (panel A) and 1,399-image quality (panel B)](Deliverables/Quality_Cost/quality_cost.png)

Panel A uses both axes from the same 100 images; panel B shows only the 1,399-image quality. Raw 100-image confusion matrices and timing records were cross-checked against archive summaries. Data points do not constitute a significance test. [Vector graphic](Deliverables/Quality_Cost/quality_cost.svg), [source data and hashes](Deliverables/Quality_Cost/figure_data.json).

**K=3 is the final method configuration; it is not the best trade-off in every scenario.** On the same 100 images, K=1 achieved both higher mIoU and faster inference than K=3. Baseline is the fastest; sliding-window achieves the highest accuracy in the main evaluation. The [final configuration decision table](Deliverables/Final_Config_Decision_TR.md) documents the rationale and different sample-size boundaries.

Checkpoint and source archive hashes, split ID lists, seeds, and the current code snapshot: [experiment manifest](Reproducibility/experiment_manifest.json) and [validation scope](Reproducibility/README_TR.md). The Git commit of historical experiments, training ID lists, and full environment dumps are not available in current sources; the current Git commit is not presented as the original version of past experiments.

## Running the Demo on Colab

1. Download the [final demo notebook](Notebooks/DINOv3_Adaptive_Zoom_Final_Demo.ipynb) and open it in Google Colab via **File → Upload notebook**.
2. Set the runtime type to **T4 GPU**.
3. Run cells in order: dependencies, restore experiment ZIP, Hugging Face login, and model loading.  
   After dependencies, run the **1b Final config** cell. The [annotated TOML](Config/DINOv3_Final_Config.toml) is embedded in the notebook; for a custom config, upload the file to `/content/DINOv3_Final_Config.toml`. K, crop size, stride, minimum entropy, and overlap threshold are read from this file. The config used is included in output ZIPs; checkpoint hash and model revision are verified.
4. Run the data cell for a VOC example. For your own JPG/PNG image, set `SOURCE="upload"` in the demo cell.
5. Run the demo cell and download the resulting output ZIP. Run the full-backup cell at the end of the session as well.

> **The required checkpoint is not included in this repository.** The experiment package containing the checkpoint — e.g., `DINOv3_complete_20261008T122013_722899Z.zip` — must be obtained separately and uploaded to the notebook. No download link is published in this repository. Cloning this repository alone does **not** provide the trained head. Your Hugging Face account must have access to the backbone used.

The default demo processes VOC index 9 with K=3. If your image has no ground truth, accuracy metrics will not be computed. The model predicts only VOC's 21 classes. This notebook is for demonstration only; it does not re-run training or large-scale benchmark experiments.

Detailed cell order and output definitions: [demo usage guide](Notebooks/Final_Demo_Kullanim_TR.md).

### Two-sample clean-session test

1. Download the [clean Colab test notebook](Notebooks/DINOv3_Clean_Colab_Test.ipynb) and upload it to Colab; select a new GPU runtime.
2. Run cells in order: **1 dependencies → 1b config → 2 checkpoint → 3 model → 4 VOC → 5 helpers → 6 test**. If setup requires a restart, continue in the same order afterward.
3. Upload your checkpoint package in step 2; complete the Hugging Face login in step 3. The embedded final config is used by default.
4. The final test cell processes indices 9 and 10 with K=3 and compares against stored reference scores within a 1e-4 tolerance. Environment versions, checkpoint/config hashes, source cell hashes, and output files are downloaded in `DINOv3_clean_smoke_*.zip`.
5. Inspect the `smoke_report.json` status. On failure, the reached stage and traceback are stored in the same ZIP. If a setup or model cell previously failed, running the test cell separately still yields a partial status report; preserve the actual error output from the failing cell separately.

### Verification status — 20 October 2026

| Check | Status |
|---|---|
| Config linking, default/custom TOML, rejection of unsupported values | Passed locally |
| Compilation of final and clean-test notebook code cells | Passed locally |
| Re-computation of 1,399-image pooled mIoU from saved matrices | Matched summary, final report, and CSV |
| Quality–cost source verification on the same 100 images | Raw records for seven methods matched summary |
| Index 9/10 test in a new Colab GPU session with the config-linked version | **20 Oct: PASS; two samples, environment/config and output hashes verified** |

In the 20 October clean test — index 9: **0.8301 → 0.9512**, index 10: **0.4658 → 0.3208**; both matched stored references. ZIP CRC, 32 output hashes, config semantics, checkpoint identity, and pipeline cell hashes were checked. Scores were independently recomputed from exported masks. Environment: Python 3.13.15, Torch 2.11.0+cu130, Transformers 5.17.0, Tesla T4. This two-image test does not imply reproducibility of training or large-scale benchmarks. [Clean Colab test report](Reproducibility/clean_colab_verification_2026-10-20_TR.md), [machine-readable evidence](Reproducibility/clean_colab_verification_2026-10-20.json).

[Saved mIoU verification report](Reproducibility/metrics_verification_TR.md) and [machine-readable report](Reproducibility/metrics_verification.json).

## Visual Examples

**Final visual package:** [four success and two failure examples](Deliverables/Final_Visuals/README_TR.md), each rendered as an 8-panel 4000×2000 px layout; plus an [Oracle–Adaptive selection reference](Deliverables/Final_Visuals/oracle_vs_adaptive_index_9.png). Panel order: input, GT, baseline, baseline error map, entropy, Top-K boxes, second-pass, and final fusion. Examples were purposively selected from the development set and do not constitute an independent evaluation. Source and mask score verification is recorded in the [visual manifest](Deliverables/Final_Visuals/selection_manifest.json).

### Improving example

VOC index 9: image mIoU **0.8301 → 0.9512**, K=3, 4 passes total.

![Index 9: baseline, entropy, selected crops, and final segmentation](Artifacts/Thesis_Figures/voc_index9_success_k3.png)

### Degrading example

VOC index 10: image mIoU **0.4658 → 0.3208**. Selecting an uncertain region does not guarantee that the local prediction will be correct.

![Index 10: example where second-pass predictions degrade the final segmentation](Artifacts/Thesis_Figures/voc_index10_failure_k3.png)

## Repository Contents

The [27 October final delivery checklist](Deliverables/DINOv3_Teslim_Kontrolu_TR.md) lists current files, numerical cross-checks, and the backup method. A [demo narration script](Deliverables/DINOv3_Final_Demo_Akisi_TR.md) is available; the [timed rehearsal log](Deliverables/DINOv3_Demo_Prova_Kaydi_TR.md) does not yet contain an actual rehearsal duration. The thesis document is a method/results section draft.

| Location | Contents |
|---|---|
| `Notebooks/` | Final Colab demo, Python helpers, and usage guide |
| `Docs/` | Final experiment report, thesis method/results draft, and roadmap documents |
| `Deliverables/` | Method comparison CSV with description, and Turkish defense presentation |
| `Artifacts/Thesis_Figures/` | Success, failure, and custom-image demo visuals |
| `Artifacts/*_review/` | Selector and refinement visual reviews |
| `Artifacts/presentation_build/` | Presentation production sources and verification outputs |

CSV: [method comparison](Deliverables/DINOv3_Method_Comparison.csv) and [column/unit description](Deliverables/DINOv3_Method_Comparison_Aciklama_TR.md). `Delta mIoU` is in percentage points, `Inference Time` in ms/image, `Relative Cost` is the time ratio relative to baseline — **not** FLOPs.

Presentation: [10-slide Turkish defense presentation](Deliverables/DINOv3_Savunma_Sunumu_TR.pptx). Thesis draft: [method and results](Docs/DINOv3_Tez_Yontem_ve_Sonuclar_Taslagi_TR.md).

`Downloads/` is excluded from Git tracking. Checkpoints and large experiment ZIPs must be kept in a separate backup. For numerical results, always refer to the current final report rather than intermediate reports.

## DINOv3 and Data Sources

- [DINOv3 technical report](https://arxiv.org/abs/2508.10104)
- [DINOv3 official repository](https://github.com/facebookresearch/dinov3)
- [VOC2012 official page](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/)

Experimental results in this repository were measured within the scope of this project and do not represent benchmark results from the DINOv3 publication. The license and access terms of the respective sources apply to model and data usage.
