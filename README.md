# Adversarial Attacks on ImageNet Classifiers

How fragile is a production-grade image classifier? This project answers that empirically, driving a pretrained **ResNet-34** from **77.40% to 0.00% top-1 accuracy** with perturbations bounded so tightly they are invisible to the eye, then testing whether those same perturbations fool a completely different architecture.

## The Setup

**Target model:** ResNet-34, pretrained on ImageNet-1K.
**Transfer model:** DenseNet-121, never used to craft any attack.
**Data:** a 500-image, 100-class subset of ImageNet-1K.

| Model | Top-1 | Top-5 |
| --- | --- | --- |
| ResNet-34 (target) | 77.40% | 93.00% |
| DenseNet-121 (transfer) | 74.00% | 92.60% |

Four attacks were implemented from scratch, three constrained to an L-infinity budget of **epsilon = 0.02** and one localised attack constrained by area instead.

## Attacks

**FGSM**, single-step. Takes one step of size epsilon in the direction of the loss gradient's sign. One forward and backward pass per image.

**PGD**, iterative. Ten steps of size 0.005, projecting back inside the epsilon-ball after each step. Effectively FGSM applied repeatedly with correction.

**MI-FGSM**, iterative with momentum. Accumulates a decaying average of past gradients (decay 0.9) with L1 normalisation at each step, so the update direction stabilises across iterations instead of oscillating. Momentum is specifically intended to improve transferability.

**Patch attack**, localised. Abandons the imperceptibility constraint entirely: instead of a tiny change everywhere, it makes a large change (epsilon 0.8) inside a single 32x32 region. Uses 40 iterations, step size 0.05, and 5 random restarts across candidate patch locations.

## Results on the Target Model

| Attack | Top-1 | Top-5 | Top-1 drop | Attack success rate | Generation time |
| --- | --- | --- | --- | --- | --- |
| None (baseline) | 77.40% | 93.00% | | | |
| FGSM | 3.40% | 20.80% | 95.6% | 82.80% | 3.9s |
| **PGD** | **0.00%** | 24.60% | **100%** | 77.40% | 49.9s |
| MI-FGSM | 0.20% | 28.60% | 99.7% | 77.20% | 51.8s |
| Patch | 12.00% | 49.40% | 84.5% | 65.40% | 925.4s |

PGD reduces top-1 accuracy to **exactly zero across all 500 images** while every pixel stays within 0.02 of its original value, a change no human observer would notice. Measured L-infinity distance sits at precisely 0.020000, confirming the constraint binds.

The single-step FGSM gets 95.6% of the way there for **one twelfth** of PGD's compute, which is the practical argument for it despite being the weakest of the three.

The patch attack is both the least effective and by far the most expensive, taking over 15 minutes because of its 40 iterations across 5 restarts per image. Its interest is not raw damage but that it operates under a fundamentally different threat model: a visible sticker rather than an invisible perturbation.

## Transferability

Each adversarial set was crafted against ResNet-34 only, then fed unchanged to DenseNet-121.

| Adversarial set | DenseNet-121 Top-1 | Top-5 | Top-1 drop from its baseline |
| --- | --- | --- | --- |
| Original (clean) | 74.00% | 92.60% | |
| **FGSM** | **45.60%** | 75.20% | **28.4 points** |
| PGD | 66.40% | 91.20% | 7.6 points |
| Patch | 67.00% | 92.00% | 7.0 points |

### The finding

**The ranking inverts.** PGD is the strongest attack on the model it was built against and the weakest when moved to another architecture. FGSM is the weakest on the source model and comfortably the best transferer, doing roughly **four times more damage** to DenseNet-121 than PGD does.

The reason is overfitting. PGD's ten corrective steps let it find a perturbation exquisitely tuned to ResNet-34's specific decision boundary, and that precision is exactly what fails to generalise. FGSM's single crude step lands on coarser, more generic features that multiple architectures happen to share.

This is the practical lesson of the project: **attack strength measured on a white-box target is a poor predictor of black-box effectiveness.** An adversary without access to the deployed model is better served by a simpler attack.

PGD and patch attacks barely dent DenseNet-121, leaving it within 8 points of its clean baseline, so neither poses much of a black-box threat here.

## Implementation

The notebook works through five tasks in sequence:

1. **Baseline evaluation.** Load the ImageNet subset via `ImageFolder`, map WordNet directory IDs to true ImageNet class indices, and establish clean accuracy for ResNet-34.
2. **FGSM.** Implement the single-step attack and generate the first adversarial set.
3. **PGD and MI-FGSM.** Implement both iterative attacks and generate a set from each.
4. **Patch attack.** Implement the localised attack with restarts and generate the third set.
5. **Transfer evaluation.** Run every adversarial set against DenseNet-121 and produce comparison plots.

Supporting utilities cover top-1 and top-5 accuracy with index mapping, L-infinity distance measurement, single-image prediction, and side-by-side visualisation of original, adversarial, and amplified perturbation.

All attacks operate in normalised image space using ImageNet channel statistics, so the epsilon budget is expressed in normalised units and the constraint is enforced against the tensor the model actually consumes.

## Tech Stack

- Python, PyTorch
- torchvision (pretrained ResNet-34 and DenseNet-121, `ImageFolder`, transforms)
- NumPy
- Matplotlib
- Pillow
- Trained and evaluated on GPU via Kaggle Notebooks

## Repository Contents

```
Deep_Learning_Project_3_Final.ipynb    Full implementation, all five tasks
Deep_Learning_Project_3_Final.html     Rendered export with outputs
Deep_Learning_Project_3_Final.pdf      PDF export
DL_Project_3_Report.pdf                Written project report
fgsm_visualization (1).png             FGSM originals, adversarials, perturbations
PGD-VIZ.png                            PGD examples
mi-fgsm viz.png                        MI-FGSM examples
patch_visualization.png                Patch attack examples
transferability_top1.png               Top-1 across models and attacks
transferability_top5.png               Top-5 across models and attacks
transfer_rates.png                     Relative transfer effectiveness
```

## Running It

```bash
git clone https://github.com/raunak-choudhary/DeepModel-AdversarialAttacks-ResNet-DenseNet.git
cd DeepModel-AdversarialAttacks-ResNet-DenseNet

python -m venv venv && source venv/bin/activate
pip install torch torchvision numpy matplotlib Pillow jupyter

jupyter notebook Deep_Learning_Project_3_Final.ipynb
```

**Dataset.** The notebook expects a `TestDataSet` directory holding the 500-image, 100-class ImageNet subset plus `labels_list.json`, which maps WordNet IDs to ImageNet class indices and readable names. It is not included here. Point `dataset_path` at your own copy.

**Paths.** Adversarial sets are written to Kaggle working paths. Adjust these for a local run.

Pretrained weights download automatically through torchvision. A GPU is recommended: the patch attack alone takes roughly 15 minutes across the 500 images.

## Authors

- **Raunak Choudhary**
- **Sharayu Rasal**

Completed for a Deep Learning course at New York University. Pretrained models from torchvision; dataset derived from ImageNet-1K.

Report preparation was assisted by Anthropic Claude 3.7 Sonnet.
