# SigBERT-MIMIC: Survival Analysis with Path Signatures

A specialized framework for survival analysis on the MIMIC-III/IV clinical dataset, leveraging **BERT** for clinical note embeddings and **Path Signatures** to capture temporal dynamics in patient trajectories.

## Project Overview
Predicting patient outcomes in the ICU requires handling both static data and irregularly sampled time-series data. This project utilizes:
- **BERT**: To extract high-dimensional semantic features from clinical observations.
- **Path Signatures (iisignature)**: To compress long-term patient paths into a compact feature set while preserving order and variation.
- **Survival Models**: Implementing Cox Proportional Hazards and Kaplan-Meier estimators via `lifelines`.

## Installation

### Prerequisites
- Anaconda or Miniconda
- Python 3.12+

### Setup
Clone this repository and install the dependencies from the `requirements.txt` file:

```bash
git clone [https://github.com/viviannnnn999/sigbert_mimic.git](https://github.com/viviannnnn999/sigbert_mimic.git)
cd sigbert_mimic
pip install -r requirements.txt
```