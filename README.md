# Federated Learning for DDoS Attack Detection in IoRT

This project implements a federated learning approach for detecting Distributed Denial of Service (DDoS) attacks in the Internet of Robotic Things (IoRT) environment. It compares the performance of different neural network architectures (CNN, LSTM, and GRU) in a federated learning setting.

## Project Overview

The Internet of Robotic Things (IoRT) integrates robotic devices with IoT infrastructure, enabling advanced perception, processing, and decision-making. This project addresses the security challenges in IoRT systems, particularly their susceptibility to DDoS attacks, by leveraging federated learning for collaborative and privacy-preserving model training.

## Project Structure

```
ddos_detection/
├── main.py
├── models.py
├── data_processing.py
├── utils.py
├── requirements.txt
└── README.md
```

- `main.py`: The main script that orchestrates the entire experiment.
- `models.py`: Contains the implementation of different neural network models (CNN, LSTM, GRU) and the federated learning training loop.
- `data_processing.py`: Includes functions for data preprocessing and client data distribution.
- `utils.py`: Utility functions for evaluation, visualization, and metrics calculation.
- `requirements.txt`: List of Python dependencies required for the project.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/dcommey/ddos-detection.git
   cd ddos-detection-federated-learning
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the CICDDoS2019 dataset:
   - The dataset should be placed in the project root directory.
   - Expected files: `ddos_data_train.csv` and `ddos_data_test.csv`

## Usage

To run the experiment, execute the main script:

```
python main.py
```

This will:
1. Load and preprocess the data
2. Train federated learning models for each architecture (CNN, LSTM, GRU)
3. Evaluate the models and generate performance metrics
4. Create visualizations for comparison

## Output

The script will generate several output files:

- Model-specific metrics and plots (for CNN, LSTM, GRU):
  - Training and validation metrics plots
  - Confusion matrices
  - ROC curves
  - F1 scores
  - Precision-recall curves
  - Training time plots
  - CSV files with training metrics
- Comparison plots and CSV files

## Contributors

- [Matilda Nkoom](https://github.com/Tilie20)
- [Daniel Commey](https://github.com/dcommey)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses the CICDDoS2019 dataset
- The federated learning implementation is inspired by the work of McMahan et al. on communication-efficient learning of deep networks from decentralized data