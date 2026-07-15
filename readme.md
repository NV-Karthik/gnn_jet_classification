# Graph Convolutional Network for Quark Gluon Jet Classification

An optimized PyTorch implementation of Dynamic Graph Convolutional Neural Network (DGCNN) for Graph Classificaition task. 

Classifying jets emerging form neutron-neutron collisions inside LHC is a long standing Deep Learning problem within HEP. 

The current particle net architecture acheives SOTA results by modelling the particle-jets as **Particle Cloud Graphs** and performing `EdgeConv` operations on them. 

Due to the sparse-nature of the dataset, the current graph-network can learn topological structures more efficiently than an image representation model.

## Enhancements

Major enhancement of this repo is the implementation of:

1. Gradient Accumulation: to accumulate gradients from micro-batch-size of 32 and achieve performance of recommended-batch-size of 384/1024.
2. Automatic Mixed Precision: BF16 for Forward/Backward passes and FP32 for weight tensors.

These optimizations helped acheive near reported accuracies on mobile GPUs with limited VRAM.

## Results

Despite a reduction in model complexity compared to the paper's baseline (366k parameters), this implementation of ParticleNet Lite scored within ~1% of the reported benchmarks.

| Task | Metric | Current Implementation (Lite - 26k) | Paper Best (Full - 366k) |
| --- | --- | --- | --- |
| **Top Tagging** | **Accuracy** | 93.07% | 94.00% |
| *(Top vs. QCD)* | **ROC AUC** | 0.9809 | 0.9858 |
| **Quark-Gluon** | **Accuracy** | 82.09% | 84.00% |
| *(w/ PID features)* | **ROC AUC** | 0.8966 | 0.9116 |

## Repo Structure

```text
├── logs/  # Training and testing logs for both benchmarks
├── saved_models/   # Serialized model weights for the best models
├── research/   # notebooks for research and experimentation
├── dataloader.py
├── dataprep.py
├── helpers.py
├── models.py
├── train.py
└── test.py
```

## Usage

Variables like batch_size, epochs, and model type (Lite vs Full) can be configured inside `train.py` and `test.py`. Once the variables are updated,

**To train the model:**

```bash
python train.py

```

**To evaluate the model:**

```bash
python test.py

```

## References & Datasets

1. **Paper:** Qu, H., & Gouskos, L. (2020). *Jet Tagging via Particle Clouds*. [arXiv:1902.08570 [hep-ph]](https://arxiv.org/abs/1902.08570).

2. **Top Tagging Dataset:** Kasieczka, G., Plehn, T., Thompson, J., & Russel, M. (2019). [Top Quark Tagging Reference Dataset](https://zenodo.org/record/2603256).

3. **Quark-Gluon Jet Classification Dataset:** Komiske, P., Metodiev, E.& Thaler, J. (2019). Pythia8 Quark and Gluon Jets for Energy Flow (Version v1) [Dataset]. Zenodo. [Link](https://doi.org/10.5281/zenodo.3164691).
   
4. P. T. Komiske, E. M. Metodiev, J. Thaler, Energy Flow Networks: Deep Sets for Particle Jets, JHEP 01 (2019) 121, [arXiv:1810.05165](https://arxiv.org/abs/1810.05165).
