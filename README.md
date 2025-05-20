# Evaluating Adversarial Attacks on ImageNet Classifiers (ResNet-34 & DenseNet-121)

This repository contains the code and resources for the project "Evaluating and Enhancing Adversarial Attacks on Deep Neural Network Image Classifiers." The project focuses on implementing various adversarial attack strategies against pre-trained image classification models (**ResNet-34** and **DenseNet-121**) on a subset of the ImageNet-1K dataset.

## Project Overview

The primary goal of this project is to "jailbreak" deep learning models by launching effective adversarial attacks to degrade their performance on image classification tasks. We explore the brittleness of these models by crafting subtle, often imperceptible, perturbations to input images that lead to misclassifications.

We implemented and evaluated the following adversarial attacks:
* **Fast Gradient Sign Method (FGSM)**: A single-step $L_{\infty}$ attack.
* **Projected Gradient Descent (PGD)**: A strong iterative $L_{\infty}$ attack.
* **Momentum Iterative FGSM (MI-FGSM)**: An iterative attack incorporating momentum for enhanced effectiveness and transferability.
* **Patch Attack**: A localized $L_0$ attack perturbing a small 32x32 region of the image.

The attacks were primarily targeted against a **ResNet-34** model, and their transferability was assessed on a **DenseNet-121** model.

## Key Findings

* **Baseline Performance (ResNet-34 on Clean Data)**:
    * Top-1 Accuracy: 77.40%
    * Top-5 Accuracy: 93.00%
* **Attack Efficacy on ResNet-34**:
    * FGSM ($\epsilon=0.02$): Top-1 3.40%, Top-5 20.80% 
    * PGD ($\epsilon=0.02$): Top-1 0.00%, Top-5 24.60%
    * MI-FGSM ($\epsilon=0.02$): Top-1 0.20%, Top-5 28.60%
    * Patch Attack (32x32, $\epsilon=0.8$): Top-1 12.00%, Top-5 49.40%
* **Transferability to DenseNet-121 (Baseline: Top-1 74.00%, Top-5 92.60%)**:
    * FGSM Adversarial: Top-1 45.60%, Top-5 75.20%
    * PGD Adversarial: Top-1 66.40%, Top-5 91.20%
    * Patch Adversarial: Top-1 67.00%, Top-5 92.00%
* Iterative attacks (PGD, MI-FGSM) were most damaging to the source model.
* Patch attacks demonstrated notable transferability.
* FGSM attacks also showed good transferability despite being simpler.

For detailed results and discussion, please refer to our project report.

## Code Implementation

The core implementation can be found in the `Deep_Learning_Project_3_Final.ipynb` Jupyter Notebook. The notebook covers the following main steps:

1.  **Setup**: Importing libraries, defining configurations (dataset paths, attack parameters), and setting up the device (CPU/GPU).
2.  **Data Loading and Preprocessing**:
    * Loading the ImageNet subset using `torchvision.datasets.ImageFolder`.
    * Applying standard ImageNet normalization and transforms.
    * Mapping dataset folder indices (WordNet IDs) to ImageNet class indices for accurate evaluation.
3.  **Model Loading**:
    * Loading pre-trained ResNet-34 (target model) and DenseNet-121 (transfer model) from `torchvision.models`.
4.  **Helper Functions**:
    * `calculate_accuracy_with_mapping` / `calculate_accuracy_adversarial`: For evaluating Top-1 and Top-5 accuracy.
    * `visualize_multiple_examples`: For displaying original images, adversarial images, and perturbations.
    * `get_prediction`: To get the top prediction for a single image.
    * `calculate_linf_distance`: To measure $L_{\infty}$ distance.
5.  **Task 1: Baseline Evaluation**: Evaluating ResNet-34 on the clean dataset.
6.  **Task 2: FGSM Attack**:
    * Implementation of `fgsm_attack` function.
    * Generation of "Adversarial Test Set 1".
    * Evaluation and visualization.
7.  **Task 3: Improved Attacks (PGD & MI-FGSM)**:
    * Implementation of `pgd_attack` function.
    * Implementation of `mi_fgsm_attack` function.
    * Generation of adversarial sets using PGD and MI-FGSM separately.
    * Evaluation and visualization.
8.  **Task 4: Patch Attack**:
    * Implementation of `advanced_patch_attack` function (32x32 patch, $\epsilon=0.8$, strategic locations, multiple restarts).
    * Generation of "Adversarial Test Set 3".
    * Evaluation and visualization.
9.  **Task 5: Transferring Attacks**:
    * Evaluating the adversarial datasets (from FGSM, PGD, Patch) on DenseNet-121.
    * Generating comparison plots for accuracies and transfer rates.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/DeepModel-AdversarialAttacks-ResNet-DenseNet.git](https://github.com/your-username/DeepModel-AdversarialAttacks-ResNet-DenseNet.git)
    cd DeepModel-AdversarialAttacks-ResNet-DenseNet
    ```
2.  **Dataset**:
    * Download the `TestDataSet` (subset of ImageNet-1K with 100 classes, 500 images, and `labels_list.json`) and place it in the root directory or update `dataset_path` in the notebook. The `labels_list.json` file contains the mapping from WordNet IDs to ImageNet class indices and human-readable names.
3.  **Environment**:
    * This project uses Python with standard data science and PyTorch libraries. Ensure you have a compatible environment. Key libraries include:
        * `torch`
        * `torchvision`
        * `numpy`
        * `matplotlib`
        * `Pillow (PIL)`
    * It is recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Linux/macOS
        # venv\Scripts\activate    # On Windows
        pip install torch torchvision numpy matplotlib Pillow jupyter
        ```
4.  **Run the Notebook**:
    * Launch Jupyter Notebook or JupyterLab:
        ```bash
        jupyter notebook Deep_Learning_Project_3_Final.ipynb
        ```
    * Execute the cells in the notebook sequentially. The notebook is structured to perform each task, generate adversarial datasets (saved as `.pt` files for images and labels, or directly as images in specified folders), evaluate models, and visualize results. Note that paths for saving adversarial datasets (e.g., `/kaggle/working/`) might need adjustment based on your local setup.

## Requirements

* Python 3.x
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* Pillow

## Authors

* Raunak Choudhary
* Sharayu Rasal

## Acknowledgments

* This project was undertaken as part of a Deep Learning course.
* Utilized pre-trained models (ResNet-34, DenseNet-121) from TorchVision.
* Dataset derived from ImageNet-1K.
* Guidance provided by the course instructors.
* Technical collaboration for report preparation: Anthropic Claude 3.7 Sonnet.

--------
