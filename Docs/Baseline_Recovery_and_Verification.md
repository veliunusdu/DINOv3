# DINOv3 baseline recovery and verification

Step 1 status on 2026-10-01: local inventory complete; original baseline assets still needed. No Colab execution or metric verification has been performed.

## Purpose

Recover and verify the baseline described in the project documents before starting Oracle Zoom. Preserve its original configuration first. A rebuilt baseline must be labelled as a new run rather than presented as a reproduction of the reported score without evidence.

## Documented baseline

| Item | Documented value |
| --- | --- |
| Backbone | Pretrained DINOv3 ViT-S/16, frozen |
| Global input | 224 x 224 RGB |
| Spatial features | 196 patch tokens reshaped to 384 x 14 x 14; CLS/register tokens excluded |
| Head | Linear / 1 x 1 convolution, 21 Pascal VOC classes |
| Output | Logits upsampled to segmentation resolution |
| Training | 5 epochs |
| Reported validation mIoU | 0.7147, not independently verified |

Sources: DINOv3_Graduation_Project_Explanation_TR.pdf, DINOv3_Ayrintili_Roadmap_TR.docx, and DINOv3_Project_Final_Progress_and_Outputs_TR.docx in this directory.

## Asset inventory

The workspace contains the three source documents. No notebook, Python implementation, trained checkpoint, dataset, run configuration, or evaluation output was found in the workspace at the time of this audit.

Locate the original Colab notebook, segmentation-head checkpoint, dataset location, and saved training/evaluation outputs. A checkpoint may reside in Google Drive even if the Colab runtime files have disappeared. Model download access must belong to the user; do not put access tokens in source files or shared notebooks.

## Recovery order

1. Obtain the original notebook or exported code and the trained head checkpoint.
2. Record the backbone identifier/revision or weights file, library versions, GPU, and checkpoint path. Preserve the original artifacts.
3. Recover the exact dataset version, training and validation split lists, preprocessing, class mapping, optimizer settings, seed, and training history where available. These are not specified completely in the documents.
4. Start a Colab GPU runtime and save experiment outputs persistently outside the temporary runtime.
5. Restore the backbone and head; verify frozen backbone parameters, evaluation mode, feature dimensions, and checkpoint compatibility.
6. Run one validation image and inspect RGB, ground truth, and prediction alignment before full evaluation.
7. Recompute validation metrics using the original protocol, then audit that protocol. If a correction changes the score, retain both results with their protocols.

## Evaluation checks

- Confirm that training and validation image identifiers are disjoint; document any augmented training data.
- Confirm that VOC labels map to classes 0 through 20 and label 255 is excluded from both loss and evaluation.
- Resize categorical masks with nearest-neighbour interpolation. Record image preprocessing and logit interpolation separately.
- Verify output channels, class order, and the exact removal of CLS/register tokens.
- Compute dataset-level per-class IoU from an accumulated confusion matrix; document treatment of absent classes and whether background is included in mIoU. Do not silently average per-image IoUs as a substitute.
- Record whether evaluation occurs at 224 x 224 or original mask resolution. Use the same evaluation grid across later methods.
- Keep original-resolution RGB images and masks accessible for Oracle Zoom; resizing an already downsampled image cannot recover detail.
- Save split identifiers, class IoUs, confusion matrix, configuration, checkpoint identity, and representative predictions alongside the aggregate score.

## Completion evidence

Step 1 is complete only when a saved notebook/code path loads the actual checkpoint, produces aligned predictions, and evaluates the recorded validation split with a documented metric protocol. The resulting mIoU must be measured, whether or not it equals 0.7147. Save the restored artifacts and results so the same model and head can be used for Oracle Zoom.

## If original assets cannot be recovered

Rebuild the documented frozen ViT-S/16 plus linear-head baseline in a new Colab notebook. Specify the missing training and evaluation choices explicitly and label all resulting scores as new measurements. Do not assume that five epochs alone will reproduce 0.7147.
