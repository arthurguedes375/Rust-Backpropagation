
use nalgebra::{DVector, DMatrix};

use crate::sigmoid::{sigmoid, sigmoid_gradient};

pub type Unit = f32;
pub type Weight = f32;

pub type Layer = DVector<Unit>;
pub type LayerWeight = DMatrix<Weight>; //actual layer X layer_before + bias

pub type Layers = Vec<Layer>;
pub type Weights = Vec<LayerWeight>;

pub enum Y {
    Indexed(DVector<Unit>), // Vector containing 1 or 2 or 3
    Matrix(DMatrix<f32>), // m X k
}

impl Y {
    pub fn val(&self) -> DMatrix<f32> {
        match &self {
            Y::Indexed(v) => {
                let m = v.row_iter().len();
                let mut result = DMatrix::<f32>::zeros(m, v.max() as usize);
                for (row, index) in v.iter().enumerate() {
                    result.row_mut(row).column_mut(*index as usize - 1)[0] = 1.0; 
                }

                result
            },
            Y::Matrix(v) => v.clone(),
        }
    }
}

pub struct Network {
    pub layers: Layers,
    pub weights: Weights,
}

pub fn forward_propagation(mut x: DMatrix<Unit>, theta: &Weights, add_bias: bool) -> (Vec<DMatrix<Unit>>, Vec<DMatrix<Unit>>) {
    if add_bias {
        x = x.insert_column(0, 1.0);
    }
    let mut layers = vec![x];
    let mut zs = vec![];

    for (i, weights) in theta.iter().enumerate() {
        let z = &layers[i] * weights.transpose();
        zs.push(z.clone());
        let sig = sigmoid(&z);
        let new_layer = if i == theta.len() - 1 {
            sig
        } else {
            sig.insert_column(0, 1.0)
        };
        layers.push(new_layer);
    }

    return (zs, layers);
}


pub fn cost_function(x: DMatrix<Unit>, theta: Weights, y: Y, lambda: f32, add_bias: bool) -> (f32, Weights) {
    let y = y.val();
    let m = x.row_iter().len() as f32;
    let mut grad = vec![];

    let (z, layers) = forward_propagation(x, &theta, add_bias);

    let log_a3 = layers[layers.len() - 1].map(|p| p.ln());
    let log_1_a3 = layers[layers.len() - 1].map(|p| (1.0 - p).ln());
    
    let first_part = (
        -y.component_mul(&log_a3) // y .* log(a3)
        -
        y.map(|b| 1.0 - b) // 1 - y
        .component_mul(&log_1_a3) // .* log(1 - a3)
    ).sum();

    
    let reg =
            (
                lambda / (2.0 * m)
            ) * theta
            .iter()
            .map(|g| g.columns_range(1..).map(|x| x.powf(2.0)).sum())
            .sum::<f32>();

    let j = first_part / m + reg;

    let last_partial = &layers[layers.len() - 1] - y;

    let mut ddelta: Weights = theta[..theta.len()].iter().map(|l| l.map(|_| 0.0)).collect();
    let range_m = m as usize;

    for i in 0..range_m {
        let d3 = last_partial.row(i).transpose();
        
        let mut partials = vec![d3];

        for lx in 1..layers.len() {
            let l = layers.len() - lx - 1;

            if l > 0 {
                let zr: DMatrix<f32> = z[l - 1].rows(i, 1).transpose();
                
                let sig = sigmoid_gradient( &zr ).insert_row(0, 1.0);

                let next_partial = &partials[lx - 1];

                let delta = (
                    theta[l].transpose() * next_partial
                ).component_mul(
                        &sig
                    );
                partials.push(delta.clone().remove_row(0));
            }

            ddelta[l] += &partials[lx - 1] * layers[l].row(i);            
        }
    }

    let first_term = lambda / m;

    for (i, d) in ddelta.iter().enumerate() {
        grad.push(d / m);
        for (c, mut v) in grad[i].column_iter_mut().enumerate() {
            if c != 0 {
                v += first_term * theta[i].column(c);
            }
        }
    }

    return (j, grad);
}