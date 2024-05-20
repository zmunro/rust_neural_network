# Rust Neural Network for Heart Disease Classification
This project implements a simple neural network in Rust to classify whether patients have heart disease based on various medical features. The neural network is trained using a dataset of patient information and their corresponding heart disease diagnosis.

## Dataset
The dataset is expected to be in a CSV file named heart_disease_data.csv with the following columns:

- `age`
- `sex`
- `chest pain type`
- `resting bp s`
- `cholesterol`
- `fasting blood sugar`
- `resting ecg`
- `max heart rate`
- `exercise angina`
- `oldpeak`
- `ST slope`
- `target` (0 or 1, where 1 indicates the presence of heart disease)

## Neural Network
The neural network consists of:

- An input layer with 11 neurons (one for each feature)
- One hidden layer with 8 neurons
- An output layer with 1 neuron (for binary classification)
- The activation function used is ReLU (Rectified Linear Unit).

## Usage
### Prerequisites
- Rust (https://www.rust-lang.org/tools/install)
- A CSV file named heart_disease_data.csv in the project directory

### Steps
1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/rust_neural_network
    cd rust_neural_network
    ```

2. Add dependencies:

    Add the following dependencies to your Cargo.toml file:

    ```toml
    [dependencies]
    ndarray = "0.15.3"
    ndarray-rand = "0.14.0"
    rand = "0.8.4"
    csv = "1.1.6"
    ```

3. Compile and run the project:

    ```sh
    cargo build
    cargo run
    ```

## Output
The program will train the neural network and calculate the ROC AUC (Receiver Operating Characteristic - Area Under the Curve) score to evaluate the model's performance. The ROC AUC score will be printed to the console.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Inspired by neural network implementations in various programming languages.
Data sourced from the UCI Machine Learning Repository.



Feel free to customize this README to better suit your project's details and requirements.







