DINOv3 Adaptive Zoom: Uncertainty-Guided Semantic Segmentation
A graduation project exploring whether uncertainty-guided local reprocessing can improve semantic segmentation while using fewer model passes than dense sliding-window inference.

This project builds on a frozen DINOv3 visual backbone and a trained semantic-segmentation head. It identifies uncertain regions in a global prediction, reprocesses selected crops at a higher effective resolution, and fuses local predictions with the original result.
Research scope: This is an experimental project, not a new foundation model. The reported numbers are project-specific measurements on Pascal VOC 2012 validation images, not benchmark figures from the DINOv3 paper.

Research question
Can we allocate additional inference computation to the most uncertain image regions instead of applying exhaustive sliding-window processing across the entire image?
Method
1. Global pass: Resize the input to 224 × 224 and obtain 21-class segmentation logits.
2. Uncertainty estimation: Compute pixelwise entropy from softmax probabilities, normalized by log(21).
3. Region selection: Rank candidate windows by mean entropy and select up to K crops using an intersection-over-union (IoU) suppression threshold.
4. Local reprocessing: Extract corresponding patches from the original RGB image, resize to the model input size, and run the same model again.
5. Logit fusion: Keep global logits outside selected regions; use locally refined logits inside them, averaging overlapping local predictions.
The selector does not use ground-truth masks at inference time.

Default final configuration: K=3, crop size 112 on the 224 × 224 selection map, stride 28, box IoU threshold 0.30, padding 0. Because of overlap suppression, fewer than K regions may be selected.
Model and experimental setup
Item	Setup
Visual backbone	Frozen facebook/dinov3-vits16-pretrain-lvd1689m
Segmentation head	21-class 1 × 1 convolution, trained for 5 epochs on the VOC2012 training split
Dataset	Pascal VOC 2012
Main evaluation	1,399 validation images, excluding a separate 50-image development group
Primary quality metric	Pooled mean Intersection over Union (mIoU), including background, with ignore label 255
Timing	100 images from the main evaluation group, measured in the same session
Environment	Google Colab with NVIDIA Tesla T4


Evaluation caveat: The main group is previously examined validation data, not a held-out untouched test set. The 50-image development group was used to choose the configuration and excluded from the main evaluation.
Results
Method	Pooled mIoU ↑	Average time (ms/image) ↓	Backbone passes
Global baseline	0.7133	79.2	1
Adaptive, K=1	0.7124	110.8	2
Adaptive, K=3 (selected configuration)	0.7220	147.5	4
Random, K=3 (mean across seeds 42/43/44)	0.7140	147.8	4 per seed
Sliding-window	0.7325	510.3	25


Key observations
- Adaptive K=3 exceeded the global baseline by 0.87 percentage points of pooled mIoU; its 95% paired-bootstrap interval was [−0.14, +1.96] points, so superiority over the baseline is not established.
- Adaptive K=3 exceeded the random K=3 strategy by 0.80 percentage points; 95% paired-bootstrap interval [+0.29, +1.33] points.
- Sliding-window had 1.05 percentage points higher mIoU than Adaptive K=3; 95% paired-bootstrap interval for Adaptive minus Sliding was [−1.51, −0.58] points.
- On the 100-image timing set, sequential, non-optimized sliding-window inference was 3.46× slower than Adaptive K=3 (95% interval [3.33×, 3.62×]). This is a measured implementation-level comparison, not a FLOPs or hardware-independent efficiency claim.
Important: K=3 was the selected final configuration, not the best option under every possible objective. In the matched 100-image quality–cost comparison, K=1 was both faster and more accurate than K=3. The baseline was fastest; sliding-window achieved the highest pooled mIoU in the main evaluation.

The three-seed random result is an average of separate runs, not an ensemble. Equal crop counts do not imply equal image-area coverage.
Qualitative analysis
Selected examples illustrate both improvements and failure modes:
- VOC index 9: Per-image mIoU improved from 0.8301 to 0.9512 with K=3.
- VOC index 10: Per-image mIoU decreased from 0.4658 to 0.3208 with K=3.
High uncertainty can indicate difficult content, but selecting an uncertain region does not guarantee that the local prediction is more accurate. The showcased examples were purposefully chosen from the development data and should not be interpreted as an independent evaluation.

See Artifacts/Thesis_Figures/ for comparison figures and Docs/ for the detailed experimental report and limitations.
Repository structure
DINOv3/
├── Artifacts/         # Example visualizations and reviewed outputs
├── Config/            # Configuration-related materials
├── Deliverables/      # Comparison tables, CSVs, and presentation materials
├── Docs/              # Experimental report, thesis draft, and supporting documentation
├── Notebooks/         # Colab demo and supporting notebook material
├── Reproducibility/   # Experiment records and validation evidence
└── README.md
Running the demo in Google Colab
1. Open the final demo notebook in Notebooks/ in Google Colab and select a T4 GPU runtime.
2. Run the dependency-installation cells, followed by the final configuration cell.
3. Upload the separately archived experiment/checkpoint ZIP when prompted.
4. Authenticate with Hugging Face if required to access the DINOv3 backbone.
5. Run model initialization and the VOC example setup.
6. Choose the default VOC example or set SOURCE="upload" to run the demo on your own JPG/PNG image.
7. Execute inference and download the resulting output ZIP.
For custom configuration, the notebook accepts a TOML file at /content/DINOv3_Final_Config.toml. The final demo validates the selected configuration and checkpoint information.

Availability limitation: The trained segmentation-head checkpoint and large experiment archives are not published in this Git repository. Therefore, cloning this repository alone is not sufficient to reproduce the trained model's outputs. The necessary checkpoint archive must be provided separately. On images without ground-truth labels, the demo produces predictions but cannot report accuracy. The model predicts only the 21 Pascal VOC classes.

See the usage guide in Notebooks/ and the reproducibility notes in Reproducibility/ for the exact notebook-cell order and verification scope.
Reproducibility and verification
The project documentation includes configuration/checkpoint hashes, archived evaluation summaries, split identifiers, seed information, and clean-session smoke-test evidence. As documented in the repository on 20 October 2026, a two-image clean Colab test reproduced stored outputs for VOC indices 9 and 10, and recorded environment/configuration information.

This does not mean the original model training or the entire large-scale benchmark was independently rerun in that smoke test. Some historical training identifiers, the original experiment's Git revision, and a complete historical environment manifest are unavailable. The current repository revision should not be represented as the exact source revision used for older experiments.
Limitations and future work
- Mixed outcomes: Local refinement can improve or worsen individual masks.
- No demonstrated baseline superiority: The paired-bootstrap interval for the K=3 versus baseline difference includes zero.
- Accuracy–latency trade-off: Sliding-window is more accurate in the main evaluation but slower in the measured implementation.
- Dataset and selection bias: Configuration selection used validation data; the main evaluation is not a previously untouched test set.
- Artifact availability: Model checkpoints are not publicly included in the repository.
Promising follow-ups include better confidence-aware fusion, improved region-selection criteria, comparisons on additional datasets, and an independently held-out evaluation.
References and attribution
- DINOv3 official repository
- DINOv3 technical report
- Pascal VOC dataset
DINOv3 and Pascal VOC were developed by their respective authors. This repository presents a separate project-specific adaptive inference pipeline and its measured experiments. Please check the original model and dataset licenses and access requirements before reuse.
Author
Veli Ünuşdu
Computer Engineering — Manisa Celal Bayar University
GitHub profile
