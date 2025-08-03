# EEG Motor Imagery Classification Using MNE and Scikit-Learn

This project implements a full pipeline for processing and classifying EEG motor imagery data using the [EEGBCI dataset](https://www.physionet.org/content/eegmmidb/1.0.0/). It leverages the MNE library for EEG data handling and preprocessing, and scikit-learn for feature extraction and machine learning classification.

## Overview

The pipeline performs the following steps:

1. **Data loading**: EEG recordings for subjects 40–89, using specific motor imagery runs.
2. **Preprocessing**: Resampling, filtering, annotation, and ICA for artifact removal.
3. **Epoching**: Segmenting the data into trials aligned to motor imagery events.
4. **Feature extraction**: Using Common Spatial Patterns (CSP).
5. **Classification**: Training and evaluating a Random Forest classifier.
6. **Visualization**: Dimensionality reduction with PCA for 2D representation of the EEG features.

## Requirements

Install dependencies using pip:

```bash
pip install numpy matplotlib seaborn scikit-learn mne
```

## Dataset

This project uses the **EEG Motor Movement/Imagery Dataset** (EEGBCI) available on PhysioNet.

To download the data using MNE:

```python
from mne.datasets import eegbci
eegbci.load_data(subject=40, runs=[4, 6, 8, 10, 12, 14])
```

Update the `data_dir` variable in the script to match the path where the data is stored. For example:

```python
data_dir = "/Users/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0"
```

## Preprocessing Details

- EEG data is resampled to 160 Hz if necessary.
- Standard 10-05 montage is applied.
- Data is filtered from 5–40 Hz.
- ICA (using FastICA) is applied to remove ocular artifacts.
- Events are mapped for motor imagery of hands and feet.
- Epochs are extracted from 0 to 2 seconds post-stimulus.

## Classification Pipeline

- CSP is used to extract spatial features from the EEG signal.
- A Random Forest classifier is trained on CSP features.
- Evaluation is performed using stratified 5-fold cross-validation.
- The best model is selected based on mean cross-validation accuracy.

## Output and Results

- Prints classification accuracy for each fold.
- Displays per-epoch predictions vs. true labels.
- Shows a 2D PCA scatter plot of the EEG features for visual inspection.

Example console output:

```
Classifier: Random Forest, Accuracy: 0.82 (+/- 0.10)
epoch 00: [1] [1] True
epoch 01: [0] [1] False
...
```

## Customization

You can modify the following parts of the script to adapt it:

| Component           | Variable / Function               | Notes                                  |
|--------------------|-----------------------------------|----------------------------------------|
| Subjects           | `subject = list(range(40, 90))`   | Adjust subject range.                  |
| Runs used          | `run_imagery = [4, 6, 8, 10, 12, 14]` | Select different runs as needed.   |
| ICA method         | `run_ica(raw, method='fastica')`  | Can be replaced with 'picard', etc.    |
| Classifier         | `classifiers` dictionary           | Add other classifiers like SVM.        |
| CSP components     | `CSP(n_components=6)`              | Tweak for feature dimensionality.      |
| Epoch rejection    | `reject=dict(eeg=60e-6)`           | Adjust based on artifact sensitivity.  |


## References

- [MNE Documentation](https://mne.tools/stable/index.html)
- [EEGBCI Dataset (PhysioNet)](https://www.physionet.org/content/eegmmidb/1.0.0/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
