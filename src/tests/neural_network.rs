use super::data::forwarded_values;

use super::super::{
    neural_network::{
        NeuralNetwork,
        weights::{ Weights, ManageWeights}
    },
    data::{
        load::Dataset,
    },
};

#[test]
fn forward_propagation() {
    let data = Dataset::from_csv_file("src/tests/data/load_csv_test.csv", Some(1.0));

    let theta = Weights::from_file("src/tests/data/nn.txt");
    let nn = NeuralNetwork::new(theta);

    let (z, layers) = forwarded_values();
    let pred = nn.forward_propagation(&data.train.x);

    for (i, zi) in pred.z.iter().enumerate() {
        assert_eq!(zi.slice_range(.., ..).to_owned(), z[i]);
    }
    for (i, li) in pred.layers.iter().enumerate() {
        assert_eq!(li.slice_range(.., ..).to_owned(), layers[i]);
    }
}