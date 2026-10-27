"""Scientific result panels from recorded predictions, without modifying predictions."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def labels_rgb(mask):
    palette=np.zeros((256,3),dtype=np.uint8)
    for label in range(256):
        value=label
        for bit in range(8):
            for channel in range(3):
                palette[label,channel] |= ((value>>channel)&1)<<(7-bit)
            value >>= 3
    palette[255]=[160,160,160]
    return palette[np.asarray(mask,dtype=np.uint8)]


def render_final_panels(image,target,baseline,entropy,regions,local_montage,final,output,source_name,scores,local_predictions=None):
    target=np.asarray(target)
    baseline=np.asarray(baseline)
    final=np.asarray(final)
    assert baseline.shape==final.shape==target.shape
    errors=np.zeros((*target.shape,3),dtype=np.uint8)
    errors[(baseline!=target)&(target!=255)]=255
    errors[target==255]=160
    fig,axes=plt.subplots(2,4,figsize=(20,10))
    axes=axes.ravel()
    axes[0].imshow(image); axes[0].set_title("Input image")
    axes[1].imshow(labels_rgb(target)); axes[1].set_title("Ground truth | gray: ignore")
    axes[2].imshow(labels_rgb(baseline)); axes[2].set_title("Baseline prediction")
    axes[3].imshow(errors); axes[3].set_title("Baseline errors | white: wrong, gray: ignore")
    heat=axes[4].imshow(entropy,cmap="inferno",vmin=0,vmax=1)
    axes[4].set_title("Normalized entropy | 224 x 224")
    fig.colorbar(heat,ax=axes[4],fraction=.046,pad=.025)
    axes[5].imshow(image); axes[5].set_title(f"Selected Top-K crops | {len(regions)}")
    for region in regions:
        x1,y1,x2,y2=region["original_box"]
        color=plt.get_cmap("tab10")((region["rank"]-1)%10)
        axes[5].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=color,linewidth=2))
        axes[5].text(x1+3,y1+3,str(region["rank"]),color=color,backgroundcolor="black",fontsize=12,va="top")
    axes[6].set_title("Second-pass predictions | crop order")
    axes[6].set_box_aspect(target.shape[0]/target.shape[1])
    if local_predictions:
        count=len(local_predictions)
        for i,pred in enumerate(local_predictions):
            child=axes[6].inset_axes([i/count,.08,1/count-.025,.78])
            child.imshow(labels_rgb(pred))
            child.set_title(f"Crop {i+1}",fontsize=11)
            child.axis("off")
    else:
        axes[6].imshow(local_montage)
    axes[7].imshow(labels_rgb(final)); axes[7].set_title("Final fused prediction")
    for ax in axes:
        ax.axis("off")
    change=100*(scores["final"]-scores["baseline"])
    fig.suptitle(f"{source_name} | K=3 | image mIoU {scores['baseline']:.4f} to {scores['final']:.4f} ({change:+.2f} pp)",fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96),h_pad=3.5,w_pad=1.3)
    fig.savefig(output,dpi=200)
    plt.close(fig)


def render_oracle_reference(image,target,entropy,adaptive_box,oracle_box,target_class,output):
    fig,axes=plt.subplots(1,4,figsize=(18,5))
    axes[0].imshow(image); axes[0].set_title("Input image")
    axes[1].imshow(labels_rgb(target)); axes[1].set_title(f"Ground truth | target class {target_class}")
    for ax,box,title,color in [(axes[2],adaptive_box,"Adaptive K=1 | entropy selection","#009A70"),
                              (axes[3],oracle_box,"Oracle | GT target box + padding","#C5A000")]:
        ax.imshow(image)
        x1,y1,x2,y2=box
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color=color,linewidth=2.5))
        ax.set_title(title)
    for ax in axes: ax.axis("off")
    fig.text(.5,.025,"Selection reference only: Oracle uses ground truth; crop areas differ. Not a matched-budget accuracy comparison.",ha="center",fontsize=11)
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(output,dpi=200)
    plt.close(fig)
