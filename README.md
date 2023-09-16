# ee109-project

# Project Report: Sequence-to-Sequence Translation on FPGA using Spatial

## 1. Introduction:
### Background:
In the era of globalization, the ability to break linguistic barriers can provide significant advantages in various fields, from commerce to culture exchange. The goal of machine translation is to enable such a feature. With the recent advances in neural networks, sequence-to-sequence models have emerged as a potent tool for this purpose.

### Objective:
The primary objective of this project is to harness the computational power of FPGA chips, known for their parallelism and customization, to execute a sequence-to-sequence model for English-to-French translation. The project uses the Spatial language, which is designed specifically for high-level synthesis targeting FPGAs.

## 2. Model Architecture and Data Processing:

### 2.1 Sequence-to-Sequence Model:

#### Design Philosophy:
The model is intentionally designed to be straightforward to ensure that the primary focus remains on FPGA integration and to minimize potential sources of error in the initial stages.

#### Model Components:

- **Input Layer:** Processes a flattened version of one-hot-encoded English sequences. Dimension: max_sequence_len x vocab_size.
- **Hidden Dense Layer:** Comprises 256 neurons and utilizes the ReLU activation function to introduce non-linearity.
- **Output Layer:** A dense layer transforming the hidden layer's outputs and applying the softmax activation function to predict each character's likelihood in the sequence.

### 2.2 Data Representation and Processing:

#### Data Source:
All input and target sequences are derived from the "input.txt" file, serving as the dataset.

#### Data Processing Pipeline:
1. Sequences are separated, tokenized, and stored.
2. A vocabulary is constructed from unique characters across all sequences.
3. Each sequence undergoes one-hot encoding, aligning with the vocabulary.
4. Data is then padded to achieve consistent sequence lengths, ensuring the model receives uniform-sized input during training.

## 3. Training, Inference, and Deployment:

### 3.1 Training in Python Environment:

#### Training Dynamics:
With the categorical cross-entropy loss and the Adam optimizer, the model is trained iteratively to reduce discrepancies between its predictions and actual translations.

#### Model Serialization:
Post training, the model's weights, vital for its decision-making ability, are serialized and stored as CSV files. This format ensures ease of use during FPGA inference.

### 3.2 Python-based Inference:

#### Procedure:
1. Data loading and preprocessing, including one-hot encoding.
2. Weight deserialization from CSV.
3. Forward propagation through the model using matrix operations and activation functions.
4. Translation sequence extraction based on model's softmax output predictions.

### 3.3 FPGA (Spatial) Inference Deployment:

#### Advantage:
FPGA offers parallel computation capabilities, enabling faster inference, especially beneficial for real-time applications.

#### Spatial Inference Workflow:
1. Transfer weights and input data from external storage (DRAM) to FPGA's local memory (SRAM).
2. Perform matrix operations, activations, and other computations in parallel using specialized functions.
3. Transfer the computed translation sequences back to DRAM.

## 5. Preliminary FPGA (Spatial) Implementation:

Although the project hasn't been validated on an FPGA, a Spatial-based structure was devised for potential deployment:

- **Data Transfer:** Model weights and input sequences were intended to be transferred between external storage (DRAM) and FPGA's local memory (SRAM).
- **Matrix Operations:** Parallel computations involving matrix operations, activations, and other necessary steps were outlined in Spatial.
- **Output Handling:** The plan was to transfer the results back to DRAM post-computation.

## 6. Challenges and Future Prospects:

### Inherent Complexities:
FPGA deployment of deep learning models involves intricate considerations, especially regarding resource management.

### Enhancements:
Advanced architectures such as LSTMs or attention mechanisms can be integrated to improve translation quality.

### Model Scaling:
While the current exploration used a simple model, scaling to commercial-grade models and their FPGA deployment will be a significant challenge and opportunity.

### FPGA Validation:
The next critical step will be the actual deployment and testing on FPGA to gather performance metrics and refine the Spatial code.

## 7. Conclusion:

This project serves as an initial investigation into FPGA-based deployment of sequence-to-sequence models for translation. Even without a finalized FPGA validation, the groundwork laid in this exploration offers a pathway for subsequent projects and studies to build upon. The intersection of deep learning with FPGA deployment remains a promising domain that promises performance advancements, especially in real-time and resource-constrained scenarios.

