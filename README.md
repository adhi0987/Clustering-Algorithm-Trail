# WEDA-FALL Dataset Processing & Modeling Pipeline

This repository contains a full end-to-end data processing and machine learning pipeline built to ingest, assemble, encode, and structurally evaluate models on the wrist-worn sensor dataset known as WEDA-FALL (Wrist Elderly Daily Activity and Fall Dataset).

## Pipeline Overview

The pipeline strictly processes the `50Hz` accelerometer, gyroscope, and orientation trial data mappings into cohesive structures for time-series modeling algorithms without user bias. It contains three core autonomous scripts executed consecutively.

### 1. `Weda-fall.py` (Dataset Assembling)
- **Goal:** Unpack and aggregate scattered sensor files into holistic participant profiles.
- **Workflow:** 
  - Dynamically extracts and parses the source dataset's markdown format to record age, gender, height, and weight for specific participant groupings.
  - Recursively sweeps through `dataset/50Hz` grabbing corresponding specific continuous files (`_accel.csv`, `_gyro.csv`, `_orientation.csv`).
  - Utilizes nearest-timestamp interpolations via `pandas.merge_asof` to synchronize 3-axis readings across separate modality streams. 
  - Attaches semantic Activity Codes & Labels (e.g., *F01* -> *Fall forward*) and saves a master data-log profile containing continuous timeline streams for each participant inside a new folder called `/Processed_Users`.

### 2. `add-labels.py` (Machine Learning Encoding)
- **Goal:** Format processing logs symmetrically strictly into fully numerical forms, bypassing textual descriptors for immediate matrix operations.
- **Workflow:**
  - Ingests the unified profiles established in step 1.
  - Automatically executes **One-Hot Encoding** configurations generating Boolean categorical dummy variables off the core structural `Activity_Code` tag.
  - Forces String categorical assignments like `Gender`, `User_ID`, and `Trial_ID` downstream into simplified boolean/numerical integers (`Male` = 0, `Female` = 1, `U01` = 1).
  - Explicitly expels string-heavy text data (like semantic Activity Names) to minimize data bloat, caching the completed structurally identical mathematical structures to a new secure path: `/Processed_Users_ML`.

### 3. `best-model.py` (Algorithm Benchmarking & Evaluations)
- **Goal:** Benchmark out-of-the-box classical modeling approaches along with modern Neural Networks to calculate specific False Positive mapping limitations evaluating activity statuses.
- **Workflow:**
  - Dynamically reverses the one-hot target classifications created earlier back into singular evaluation index thresholds mapping to `y`.
  - Intelligently ignores any metadata artifacts relating back to the physical identity (e.g., dropping Age, Gender, specific times, ID tracking) isolating testing entirely upon the 9 dimensional ranges of pure physical X, Y, Z movements.
  - Executes comprehensive *StandardScaling* array transformations to prevent structural variance differences across measurements crushing back-propagation methodologies.
  - Trains matrices on an `80/20` train/test ratio *within the participant's independent bounds*, feeding structures sequentially against identical iterations initialized in:
    - **Random Forest** (n=10)
    - **Decision Tree**
    - **Logistic Regression** (max_iter=200)
    - **Multi-layer Perceptron (Neural Network)** (Hidden: 64, max_iter=200)
  - Exports holistic `Accuracy` maps arrayed alongside `Macro Averaged False Positive Rates` out dynamically to standard CSV data logs (`model_accuracies.csv`, `model_fprs.csv`) and a synchronized terminal footprint inside `output.log`.

## Usage
Execute scripts in terminal precisely sequential:
```bash
python Weda-fall.py
python add-labels.py
python best-model.py
```

## Conclusions Output
Generally upon executing `best-model.py`, **Random Forest Model** combinations consistently override classic logistic boundaries tracking at higher ~98% predictive accuracies alongside minute sub-0.005 macro-FPR metrics, acting natively resistant to multi-sensory boundary overlaps.
