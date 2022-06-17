use criterion::{Criterion, criterion_main, criterion_group, black_box};
use neural_network::data::load::load_csv;

pub fn benchmark(c: &mut Criterion) {
    c.bench_function("load_csv", |b| b.iter(|| {
        load_csv("benches/test.csv", None, false);
    }));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark,
);
criterion_main!(benches);