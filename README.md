# EPF using AV-C-PID
Probabilistic Electricity Price Forecasting (PEPF) in the Irish Day-Ahead Market (DAM) using Conformal Prediction methods, including the Asymmetric Volatility-Aware Conformal PID (AV-C-PID) extension.

## Project Setup

1. **Install Python 3.10**: Ensure you have Python 3.10 installed. Download from [python.org](https://www.python.org/downloads/).
2. **Install Anaconda**: Download and install Anaconda from [anaconda.com](https://www.anaconda.com/products/individual).

## Virtual Environment Setup

1. **Create a virtual environment**:
   ```bash
   conda create --name epf_env python=3.10
   ```

2. **Activate the environment**:
   ```bash
   conda activate epf_env
   ```

3. **Navigate to the project folder**:
   ```bash
   cd /path/to/epf_v2
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Reproduce Results

1. **Place the data file** `DAM_data.csv` in the project root directory.

2. **Create an empty folder** `results/` in the prject root directory.

3. **Make changes in configurations** using `configs.py`.
   
4. **Run the full DAM pipeline**:
   ```bash
   python run_dam.py
   ```
   This trains all four base models (KNN, LEAR, RF, LGBM) and applies all nine PI estimation methods sequentially. Prediction CSV files are saved incrementally to `results/` as each method completes.

5. **If the run is interrupted**, resume from the last completed method without recomputing finished results:
   ```bash
   python resume_dam.py
   ```

6. **Evaluation metrics** (APS, IW, coverage, Winkler) are computed automatically at the end of the run and saved to `DAM_metrics.csv`.

7. **Trading results** (TS1 and TS2 battery strategies) are computed automatically at the end of the run and saved to `DAM_trading.csv`.

8. **Reproduce the volatility analysis figures**:
   ```bash
   python volatility_analysis.py
   ```
   This processes all CSV files in `results/` automatically and saves timeline and decile plots alongside each file.

9. **Inspect individual result files** at any point during the run:
   ```bash
   python check_coverage.py
   ```
   Edit `FILE_NAME` at the top of `check_coverage.py` to select any result file.

> **Note**: The full DAM pipeline on a MacBook Air M1 (macOS Sequoia 15.1, 8 GB RAM) takes approximately 300 hours across 365 daily retraining steps. Results for individual methods can be inspected as they are saved incrementally to `results/`.

## References

This project replicates and extends the methodology of O'Connor, Collins, Prestwich, and Visentin (2025).

O'Connor, C., Collins, J., Prestwich, S., and Visentin, A. (2025). Conformal Prediction for Electricity Price Forecasting in Day-Ahead and Balancing Markets. *Energy and AI*, 21, 100571.
https://doi.org/10.1016/j.egyai.2025.100571

The AV-C-PID method extends the conformal PID control framework of Angelopoulos, Bates, Malik, and Jordan (2023).

Angelopoulos, A., Bates, S., Malik, J., and Jordan, M. (2023). Conformal PID Control for Time Series Prediction. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
https://arxiv.org/abs/2307.16895
