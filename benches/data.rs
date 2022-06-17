use criterion::{Criterion, criterion_main, criterion_group, black_box};
use neural_network::{data::{load::Dataset, ManageX, ManageY}, neural_network::weights::{Weights, ManageWeights}};

pub fn benchmark(c: &mut Criterion) {
    let data = Dataset::from_csv_file("benches/test.csv", None);
    let content = black_box(std::fs::read_to_string("benches/theta_test.txt").unwrap());
    let csv_content = black_box(std::fs::read_to_string("benches/test.csv").unwrap());

    c.bench_function("Dataset::from_csv_string", |b| b.iter(|| {
        Dataset::from_csv_string(&csv_content, None);
    }));
    c.bench_function("Weights::init", |b| b.iter(|| {
        let hidden_layers = 20;
        let hidden_layers_len = 500;
        Weights::init(
            data.train.x.input_nodes(),
            hidden_layers,
            hidden_layers_len,
            data.train.y.output_classes(),
            None,
        );
    }));
    c.bench_function("Weights::from_string", |b| b.iter(|| {
        Weights::from_string(&content);
    }));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark,
);
criterion_main!(benches);