extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate csv;

use std::cmp::Ordering;
use std::error::Error;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;


fn calculate_roc_auc(predictions: &Array2<f64>, targets: &Array2<f64>) -> f64 {
    let mut data: Vec<(f64, f64)> = predictions.iter().zip(targets.iter()).map(|(p, t)| (*p, *t)).collect();
    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut tp_prev = 0.0;
    let mut fp_prev = 0.0;
    let mut auc = 0.0;

    for i in 0..data.len() {
        if data[i].1 == 1.0 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        if i == data.len() - 1 || data[i].0 != data[i + 1].0 {
            auc += (fp - fp_prev) * (tp + tp_prev) / 2.0;
            tp_prev = tp;
            fp_prev = fp;
        }
    }

    auc / (tp * fp)
}

struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    learning_rate: f64,
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = StdRng::seed_from_u64(42);
        let weights_input_hidden = Array::random_using((input_size, hidden_size), Uniform::new(-1.0, 1.0), &mut rng);
        let weights_hidden_output = Array::random_using((hidden_size, output_size), Uniform::new(-1.0, 1.0), &mut rng);
        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            learning_rate,
            weights_input_hidden,
            weights_hidden_output,
        }
    }

    fn relu(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|x| x.max(0.0))
    }

    fn relu_derivative(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn forward(&self, inputs: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {

       let hidden_inputs = inputs.dot(&self.weights_input_hidden);
       let hidden_outputs = Self::relu(&hidden_inputs);

       let final_inputs = hidden_outputs.dot(&self.weights_hidden_output);
       let final_outputs = Self::relu(&final_inputs);
       (hidden_outputs, final_outputs)
    }

    fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>) {
        let (hidden_outputs, final_outputs) = self.forward(inputs);
        let output_errors = targets - final_outputs.clone();
        let hidden_erros = output_errors.dot(&self.weights_hidden_output.t());

        let delta_weights_hidden_output = self.learning_rate * hidden_outputs.t().dot(&(output_errors * &Self::relu_derivative(&final_outputs)));
        let delta_weights_input_hidden = self.learning_rate * inputs.t().dot(&(hidden_erros * &Self::relu_derivative(&hidden_outputs)));

        self.weights_hidden_output += &delta_weights_hidden_output;
        self.weights_input_hidden += &delta_weights_input_hidden;
    }

    fn predict(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let (_, final_outputs) = self.forward(inputs);
        final_outputs
    }
}

fn load_data(filename: &str) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(filename)?;
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for result in reader.records() {
        let record = result?;
        let record: Vec<f64> = record.iter().map(|s| s.parse().unwrap()).collect();
        inputs.push(record[..11].to_vec());
        targets.push(vec![record[11]]);
    }

    let inputs = Array2::from_shape_vec((inputs.len(), 11), inputs.into_iter().flatten().collect())?;
    let targets = Array2::from_shape_vec((targets.len(), 1), targets.into_iter().flatten().collect())?;
    Ok((inputs, targets))
}

fn main() -> Result<(), Box<dyn Error>>{
    let (inputs, targets) = load_data("data.csv")?;

    let input_size = 11;
    let hidden_size = 8;
    let output_size = 1;
    let learning_rate = 0.01;

    let mut nn = NeuralNetwork::new(input_size, hidden_size, output_size, learning_rate);

    for _ in 0..1000 {
        for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
            let input = input.to_owned().insert_axis(Axis(0));
            let target = target.to_owned().insert_axis(Axis(0));
            nn.train(&input, &target);
        }
    }

    let mut predictions = Array2::<f64>::zeros((inputs.shape()[0], 1));
    for (i, input) in inputs.outer_iter().enumerate() {
        let input = input.to_owned().insert_axis(Axis(0));
        predictions.row_mut(i).assign(&nn.predict(&input).into_shape((1,)).unwrap());
    }

    let roc_auc = calculate_roc_auc(&predictions, &targets);
    println!("ROC AUC: {}", roc_auc);

    Ok(())
}
