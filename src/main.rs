extern crate nalgebra as na;
extern crate rand;
extern crate csv;

use std::sync::mpsc::channel;

pub mod sigmoid;
pub mod neural_network;
pub mod data;
pub mod task;

fn main() {
    let (tx, rx) = channel();
    ctrlc::set_handler(move || tx.send(()).unwrap())
        .expect("Error setting Ctrl-C handler");

    let data = data::load::load_csv("datasets/train.csv");
    let weights = data::Weights::load("weights/1655134914251.txt");

    let lambda = 0.001;

    let mut network = neural_network::NeuralNetwork::new(weights);
    network.train(&data.x, &data.y, 0.05, lambda, 10, 200, true, rx);
    println!("{}", network.precision(&data.x, &data.y));
}
