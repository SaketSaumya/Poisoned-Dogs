"""Poison Frogs replication: targeted clean-label poisoning (Shafahi et al., NeurIPS 2018).

Modules, in the order the project uses them:
    config      paths and default settings
    images      loading, saving and comparing images
    data        building the datasets from the archive
    features    frozen ResNet-50 feature extractor + trainable last layer (part 1)
    attack      Algorithm 1 (feature collision) and watermarking
    transfer    part 1 experiments: the one-shot kill on a frozen network
    e2e         part 2 experiments: end-to-end retraining of a small AlexNet
    plots       static figures
    animations  GIFs
    gif         flicker-free GIF writer
"""
