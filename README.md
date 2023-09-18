# ee109-project: Sequence-to-Sequence Translation using Spatial

## 1. Introduction:
### Background:
In the era of globalization, the ability to break linguistic barriers can provide significant advantages in various fields, from commerce to culture exchange. The goal of machine translation is to enable such a feature. With the recent advances in neural networks, sequence-to-sequence models have emerged as a potent tool for this purpose.

### Objective:
The primary objective of this project is to harness the computational power of FPGA chips, known for their parallelism and customization, to execute a sequence-to-sequence model for English-to-French translation. The project uses the Spatial language, which is designed specifically for high-level synthesis targeting FPGAs.

### To recreate:
On running the python code, download the csv weight files and also take note of the vocabSize and maxTextLength reported in the .ipynb file and change those variables accordingly in the spatial file. Then, to run inference in spatial, choose an english sentence and first run it in the python inference code. Then, you will get the python french translation and also download the input_one_hot_flattened.csv file from the .ipynb notebook. Then, upload all csv files accordingly to the same spatial directory your code is written in, and run it. 

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
All input and target sequences are derived from the "input.txt" file, serving as the dataset with english and french sentences formatted like: 
I eat an apple.*Je mange une pomme.
Where the english sentence is seperated from the french translation with an asterisk. Furthermore, the current input.txt has only around 300 samples which is very small, and has a lot of potential to be increased to potentially produce better results. 

#### Data Processing Pipeline:
1. Sequences are separated, tokenized, and stored.
2. A vocabulary is constructed from unique characters across all sequences.
3. Each sequence undergoes one-hot encoding, aligning with the vocabulary.
4. Data is then padded to achieve consistent sequence lengths, ensuring the model receives uniform-sized input during training.

## 3. Training, Inference, and Deployment:

### 3.1 Training in Python Environment:

![alt text](https://github.com/guananya/ee109-project/blob/main/img/trainGraph.png?raw=true)

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

![alt text](https://github.com/guananya/ee109-project/blob/main/img/ee109-blockdiag.jpg?raw=true)

### 5.1 Data Representation and Transfer:

#### Fixed-Point Arithmetic:
Neural networks' computational requirements and FPGAs' inherent nature necessitate a shift from the commonly used floating-point arithmetic to fixed-point arithmetic. This transition aids in achieving better resource utilization and power efficiency. In our implementation, we leveraged 16-bit fixed-point numbers, providing a trade-off between precision and resource consumption.

#### Data Transfer:
The mechanisms to transfer data, including model weights, biases, and input sequences, between DRAM and the FPGA's SRAM were envisioned. Efficient transfer is crucial, given the vast difference in access times between the two memory types.

### 5.2 Matrix Operations and Parallelism:

#### Parallelism with foreach:
The Spatial language provides looping constructs like `foreach` that enable parallelism on FPGAs. Our model's matrix operations, particularly during forward propagation, were optimized using `foreach` loops. This led to simultaneous processing of multiple neurons or data points, significantly boosting computational efficiency.

#### Matrix Multiplication:
This operation, being the heart of neural network computations, was optimized to capitalize on FPGA's spatial architecture. Using pipelining and parallel processing, matrix multiplication was accelerated, ensuring that multiple operations occurred concurrently.

#### Activation Functions:
Non-linear activation functions, critical to neural networks' success, were implemented in fixed-point arithmetic. Their lookup tables were stored in the SRAM for faster access.

### 5.3 Output Handling and Data Transition:

#### Buffering and Data Flow:
Intermediate results during forward propagation were buffered in the SRAM to reduce frequent and costly data transfers to DRAM.

#### Output Alchemy:
Once computations were completed, a mechanism was put in place to transition the results back to DRAM. These results, stored as fixed-point numbers, were then converted back to a more human-friendly format for interpretation and validation.

## 6. Challenges and Future Prospects:

### Precision vs. Efficiency:
Fixed-point arithmetic, while efficient, brings challenges related to precision. Ensuring adequate precision without substantial resource overhead remains an ongoing challenge.

### Loop Unrolling with foreach:
While `foreach` brings parallelism, judicious loop unrolling is crucial. Excessive unrolling can consume FPGA resources rapidly, while insufficient unrolling might not fully exploit the FPGA's potential.

### Advanced Architectures:
Implementing more intricate architectures like LSTMs or attention mechanisms on FPGA poses challenges due to their increased complexity and memory requirements.

### Benchmarking:
Given the theoretical implementation, future work involves real-world testing to determine computational efficiency, speedup, and model accuracy on the FPGA.

## 7. Conclusion:

This project serves as an initial investigation into FPGA-based deployment of sequence-to-sequence models for translation. Even without a finalized FPGA validation, the groundwork laid in this exploration offers a pathway for subsequent projects and studies to build upon. The intersection of deep learning with FPGA deployment remains a promising domain that promises performance advancements, especially in real-time and resource-constrained scenarios.

