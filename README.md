# Time-Dependent Segmentation of Electron Microscopy Images Using Deep Learning and Image Processing Methods

This repository is associated with the MSc project of Mobina Azimi at Forschungszentrum Jülich (FZJ).
The objective of this work is to develop, analyze, and compare time-dependent segmentation approaches for electron microscopy (EM) image sequences, including real and synthetically generated datasets.


## Project Overview

Electron microscopy provides high-resolution visualizations of nanoscale structures.  
When objects change **position, shape, or intensity across time**, segmentation becomes a difficult problem due to:

- Low signal-to-noise ratio  
- Variability in object texture and contrast  
- Temporal motion and deformation  
- Lack of ground-truth annotation across entire sequences  

This project evaluates **two complementary segmentation strategies**:

| Method | Approach | Strengths |
|--------|----------|-----------|
| **MGAC (Morphological Geodesic Active Contours)** | Shape-based contour evolution guided by edges and intensity | Stable tracking, interpretable and classical |
| **CNN–ConvLSTM–UNet Model** | Spatiotemporal deep learning encoder-decoder | Learns motion + structural evolution over time |

Additionally, a **synthetic EM data generator** is included to create controlled datasets for benchmarking and validation.
