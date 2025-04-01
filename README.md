# Image Stitching for NSOM Microscopy

The **stitching** repository, developed as part of research in the RBNI Lab at the Electrical Engineering Faculty, Technion, provides Python scripts for stitching images obtained from Near-field Scanning Optical Microscopy (NSOM).

NSOM is a high-resolution imaging technique used for capturing nanoscale surface details. However, due to the high magnification, the images typically suffer from **low Signal-to-Noise Ratio (SNR)** and have **very few common features** between adjacent frames.

To overcome these challenges, the algorithm follows a multi-step process:

1. **Preprocessing for Noise Reduction**:

   - Since NSOM images often contain artifacts from lens contamination, a **precomputed average from multiple images** is used as a **mask** to filter out dirt and unwanted patterns.
   - The algorithm applies a **blurring filter** to suppress background noise and a **sharpening filter** to enhance prominent features in the images.

2. **Feature Matching and Camera Position Estimation**:

   - After filtering, only a minimal set of distinguishable features remains.
   - A **standard feature-matching algorithm** is applied to identify correspondences between overlapping image regions.
   - The algorithm **filters matches based on prior knowledge of the camera’s movement relative to the scanned sample** and **removes outliers using statistical methods**.

3. **Final Image Stitching**:

   - The aligned images are stitched together to produce a seamless, high-resolution composite.

### Dependencies

Ensure you have the required Python libraries installed before running the scripts:

```bash
pip install opencv-python numpy pillow
```

##
