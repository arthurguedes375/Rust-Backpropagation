use criterion::{Criterion, criterion_main, criterion_group, black_box};
use neural_network::data::{load::load_csv, Weights};

pub fn benchmark(c: &mut Criterion) {
    let data = load_csv("benches/test.csv", None, false);
    c.bench_function("load_csv", |b| b.iter(|| {
        load_csv("benches/test.csv", None, false);
    }));
    c.bench_function("init_theta", |b| b.iter(|| {
        let hidden_layers = 20;
        let hidden_layers_len = 500;
        Weights::init(
            &data.train.x,
            hidden_layers,
            hidden_layers_len,
            &data.train.y,
            None,
            false
        );
    }));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark,
);
criterion_main!(benches);