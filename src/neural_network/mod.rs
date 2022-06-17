pub mod types;
pub mod weights;
use weights::Weights;

use std::sync::{mpsc::{channel, Receiver}, Arc};
use std::io::{self, Write};
use crate::task::{init_task, end_task, is_debugging};
use na::{DMatrix};
use types::{Unit};
use crate::data::{X, Y};


use crate::sigmoid::{sigmoid, sigmoid_gradient};


pub struct PropagatedNetwork {
    pub z: Vec<Arc<DMatrix<Unit>>>,
    pub layers: Vec<Arc<DMatrix<Unit>>>,
}
pub struct Evaluation {
    pub cost: f32,
    pub gradient: Weights,
}

#[derive(Clone)]
pub struct NeuralNetwork {
    pub theta: Weights,
}

impl NeuralNetwork {
    pub fn new(theta: Weights) -> NeuralNetwork {
        NeuralNetwork {
            theta,
        }
    }

    pub fn set_ctrlc_handler() -> Receiver<()> {
        let (tx, rx) = channel();
        ctrlc::set_handler(move || tx.send(()).unwrap())
            .expect("Error setting Ctrl-C handler");

        return rx;
    }

    pub fn forward_propagation(&self, x: &X) -> PropagatedNetwork {
        let mut layers = vec![x.clone()];
        let mut zs = vec![];
    
        for (i, weights) in self.theta.iter().enumerate() {
            let z = &*layers[i] * weights.transpose();
            let sig = sigmoid(&z);
            zs.push(Arc::new(z));
            let new_layer = if i == self.theta.len() - 1 {
                sig
            } else {
                sig.insert_column(0, 1.0)
            };
            layers.push(Arc::new(new_layer));
        }
    
        return PropagatedNetwork {
            z: zs,
            layers,
        };
    }

    pub fn precision(&self, x: &X, y: &Y) -> f32 {
        init_task("Precision: Forward Prop");
        let results = self.forward_propagation(x);
        end_task();

        init_task("Precision: Calculating");
        let mut right = 0;
        for (i, r) in results.layers[results.layers.len() - 1].row_iter().enumerate() {
            let mut max_ri = 0;
            let mut found_y = false;

            let max_rv = r.max();

            let mut max_yi = 0;
            for (i, v) in y.row(i).iter().enumerate() {
                if *v == 1.0 {
                    max_yi = i;
                    found_y = true;
                }

                if found_y && r[max_yi] == max_rv {
                    max_ri = i;
                    break;
                }
            } 
            
            if max_ri == max_yi {
                right += 1;
            }
        }
        end_task();

        return right as f32 * 100.0 / y.nrows() as f32;
    }

    pub fn cost_function(&self, x: &X, y: &Y, lambda: f32) -> Evaluation {
        let m = x.row_iter().len() as f32;
        let mut grad = vec![];
    
        let PropagatedNetwork {
            z,
            layers,
        } = self.forward_propagation(&x);
    
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
                ) * self.theta
                .iter()
                .map(|g| g.columns_range(1..).map(|x| x.powf(2.0)).sum())
                .sum::<f32>();
    
        let j = first_part / m + reg;
    
        let last_partial = &*layers[layers.len() - 1] - &**y;
    
        let mut ddelta: Weights = self.theta[..self.theta.len()].iter().map(|l| l.map(|_| 0.0)).collect();
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
                        self.theta[l].transpose() * next_partial
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
                    v += first_term * self.theta[i].column(c);
                }
            }
        }
    
        return Evaluation {
            cost: j,
            gradient: grad,
        };
    }

    pub fn train(
        &mut self,
        x: &X,
        y: &Y,
        alpha: f32,
        lambda: f32,
        iters: usize,
        batch_size: usize,
        rx: Receiver<()>
    ) {
        if is_debugging() { println!("Training... "); }
        let mut batches: Vec<[Arc<DMatrix<f32>>; 2]> = vec![];

        for i in 0..iters {
            if let Ok(_) = rx.try_recv() {
                break;
            }
            let v = i % (x.nrows() / batch_size);
            let v1 = v * batch_size;
            let v2 = v * batch_size + batch_size;
            if batches.len() <= v {
                let batch_x = x
                    .slice_range((v1)..(v2), ..).clone_owned();
                let batch_y = y
                    .slice_range((v1)..(v2), ..).clone_owned();

                batches.push([Arc::new(batch_x), Arc::new(batch_y)]);
            }
            
            let evaluation = self.cost_function(&batches[v][0], &batches[v][1], lambda);
            for (i, grad) in evaluation.gradient.iter().enumerate() {
                self.theta[i] -= grad * alpha;
            }
            if is_debugging() {
                print!("\rIteration: {i}, Mini batch: {v1}..{v2}, Cost: {}\r", evaluation.cost);
                io::stdout().flush().unwrap();
            }
        }
    }
}



