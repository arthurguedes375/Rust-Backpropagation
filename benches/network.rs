use criterion::{criterion_group, criterion_main, black_box, Criterion};

use neural_network::{   
    neural_network::{
        weights::{Weights, ManageWeights},
        NeuralNetwork,
    },
    data::load::Dataset,
};

pub fn benchmark(c: &mut Criterion) {
    let theta = black_box(Weights::from_file("benches/nn.txt"));
    let x = black_box(Dataset::from_csv_file("benches/test.csv", Some(1.0)));
    let net = black_box(NeuralNetwork::new(theta));
    c.bench_function("Network::forward_propagation", |b| b.iter(|| {
        net.forward_propagation(&x.train.x);
    }));
}

criterion_group!(
    name=benches;
    config = Criterion::default();
    targets=benchmark,
);
criterion_main!(benches);
