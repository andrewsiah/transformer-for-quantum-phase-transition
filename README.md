# Quantum Phase Transition Detection using Self-Supervised Learning

Thesis Paper: https://www.overleaf.com/read/bsrbjzdbrbhr#5dec37

## Abstract
This project explores a novel deep learning approach to identify quantum phase transitions (QPTs) in quantum many-body systems. We employ self-supervised machine learning on a decoder-only sequence model architecture, specifically a Decoder Transformer. The data is derived from reduced density matrix (RDM) data using Exact Diagonalization (ED) across the range $\Delta = [-2,2]$. Our method effectively identifies the quantum phase transition between Antiferromagnetic (AFM) and XY phases at $\Delta = -1$. This work builds upon previous groundbreaking research utilizing advanced deep learning techniques in quantum phase transition, extending the work of Dr. Wing Chi Yu and colleagues. For more details, refer to the following publications.

## Data Generation
The data generation script is located in `data_gen_xxz_s_half.py`. To generate the data, follow these steps:
1. Navigate to the project directory: `cd dissert`
2. Start a new screen session: `screen`
3. Run the data generation script: `python data_gen_xxz_s_half.py`

## Modules
### `gpt2.py`
This module contains the implementation of GPT-2, which is used to detect the first quantum phase transition within the range $\Delta = [-2,2]$. The model is trained to identify the transition effectively.

## Environment Setup
The current environment that works for training is managed using `poetry`. Ensure you have `poetry` installed and configured to replicate the training environment.

For any further questions or issues, please refer to the documentation or open an issue on the GitHub repository.