pub mod load;

use na::{DVector, DMatrix};
use rand::Rng;
use crate::task::{init_task, end_task};
use crate::neural_network::types::{Unit, Weights as Ws};

pub enum Y {
    Indexed(DVector<Unit>), // Vector containing 1 or 2 or 3
    Matrix(DMatrix<f32>), // m X k
}

impl Y {
    pub fn val(&self) -> DMatrix<f32> {
        match &self {
            Y::Indexed(v) => {
                let m = v.row_iter().len();
                let mut result = DMatrix::<f32>::zeros(m, v.max() as usize + 1);
                for (row, index) in v.iter().enumerate() {
                    result.row_mut(row).column_mut(*index as usize)[0] = 1.0; 
                }

                result
            },
            Y::Matrix(v) => v.clone(),
        }
    }
}

pub enum X {
    Biased(DMatrix<Unit>), // Already has bias unit
    NotBiased(DMatrix<Unit>), // Does not have bias unit
}

impl X {
    pub fn val(&self) -> DMatrix<f32> {
        match &self {
            X::NotBiased(x) => {
                x.clone().insert_column(0, 1.0)
            },
            X::Biased(x) => x.clone(),
        }
    }
}

pub struct Weights {}

impl Weights {
    pub fn init(x: &X, hidden_layers: u16, hidden_layers_len: u16, y: &Y, epsilon: f32) -> Ws { 
        let x = x.val();
        let y = y.val();
        println!(
"========
Training examples: {}
Output layers: {}
Hidden Layers: {hidden_layers},
Hidden layers len: {hidden_layers_len}
========
", x.nrows(), y.ncols()
);
        init_task("Initializing theta");
        let mut rand = rand::thread_rng();
        let theta_len = hidden_layers + 1;

        let mut weights: Ws = vec![];

        for i in 0..theta_len {
            let previous_len = if i == 0 {
                x.ncols()
            } else {
                hidden_layers_len as usize + 1
            };

            let next_len = if i == theta_len - 1 {
                y.ncols()
            } else {
                hidden_layers_len as usize
            };

            let init_rand = |_, _| {
                rand.gen_range(-epsilon..epsilon)
            };

            let w = DMatrix::from_fn(
                next_len,
                previous_len,
                init_rand,
            );

            weights.push(w);
        }

        end_task();
        return weights;
    }

    pub fn load(path: &str) -> Ws {
        init_task("Loading Weights");
        let file = std::fs::read_to_string(path).unwrap();
        end_task();

        init_task("Processing Weights");
        let mut weights = vec![];

        let layers = file.split('=').collect::<Vec<&str>>();

        for layer in layers {
            let rows = layer.split('\n').collect::<Vec<&str>>();
            let rows = rows[..rows.len()-1].to_vec();

            let columns = rows[0].matches(",").count();

            let mut nw = DMatrix::<f32>::zeros(
                rows.len(),
                columns
            );

            for (r, row) in rows.iter().enumerate() {
                let columns = row.split(',').collect::<Vec<&str>>();
                let columns = columns[..columns.len() - 1].to_vec();
                for (c, column) in columns.iter().enumerate() {
                    nw.row_mut(r).column_mut(c)[0] = column.parse().unwrap();
                }
            }

            weights.push(nw);
        }

        end_task();
        return weights;
    }
}