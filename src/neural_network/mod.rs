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
pub type PropagatedLayer = Vec<Arc<DMatrix<Unit>>>;

pub struct PropagatedNetwork {
    pub z: PropagatedLayer,
    pub layers: PropagatedLayer,
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

    /*
     * Sets the ctrlc_handler and returns a channel
     * that will receive a message if the app gets an unexpected ctrlc
     */
    pub fn set_ctrlc_handler() -> Receiver<()> {
        let (tx, rx) = channel();
        ctrlc::set_handler(move || tx.send(()).unwrap())
            .expect("Error setting Ctrl-C handler");

        return rx;
    }

    pub fn forward_propagation(&self, x: &X) -> PropagatedNetwork {
        // Initializes the layers vector and sets the first layer to be the input 
        let mut layers = vec![x.clone()];

        // Initializes the z vector
        let mut zs = vec![];
    
        // Loops through every theta layer
        for (i, weights) in self.theta.iter().enumerate() {

            // Multiplies the current layer values with the respective weights to get z
            let z = &*layers[i] * weights.transpose();

            // Runs the activation function on the multiplied value 
            let sig = sigmoid(&z);

            // Save current z so it can be used to generate the partial derivatives later on
            zs.push(Arc::new(z));

            // If it is not the last layer then it adds the bias unit to the activated layer
            let new_layer = if i == self.theta.len() - 1 {
                sig
            } else {
                sig.insert_column(0, 1.0)
            };

            // Save the layer's result
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

        /*
         * Runs through every result
         * Then it gets the value of the strongest fired neuron
         * Then it figures out the index of the right prediction
         * Then it checks if the value strongest fired neuron is the same as the index of the right prediction
         */ 
        for (i, r) in results.layers[results.layers.len() - 1].row_iter().enumerate() {
            // Gets the greatest neuron's value
            let max_fired_value = r.max();

            for (right_index, predicted) in y.row(i).iter().enumerate() {
                if *predicted == 1.0 {
                    if r[right_index] == max_fired_value {
                        right += 1;
                    }
                    break;
                }
            }
        }
        end_task();

        return right as f32 * 100.0 / y.nrows() as f32;
    }

    pub fn get_cost(&self, layers: PropagatedLayer, y: &Y, lambda: f32) -> f32 {
        // Number of training examples
        let m = y.nrows() as f32;
        
        // log(propagated_class)
        let log_a3 = layers[layers.len() - 1].map(|p| p.ln());
        
        // log(1 - propagated_class)
        let log_1_a3 = layers[layers.len() - 1].map(|p| (1.0 - p).ln());
        
        // Not regularized term
        let first_part = (
            -y.component_mul(&log_a3) // y .* log(a3)
            -
            y.map(|b| 1.0 - b) // 1 - y
            .component_mul(&log_1_a3) // .* log(1 - a3)
        ).sum();
    
        // Adding regularization term
        let reg =
                (
                    lambda / (2.0 * m)
                ) * self.theta
                .iter()
                .map(|g| {
                    g.columns_range(1..).map(|x| x.powf(2.0)).sum() // sum(theta[bias_unit + 1..] .^ 2)
                })
                .sum::<f32>();
    
        // First part + regularization
        let j = first_part / m + reg;

        return j;
    }

    pub fn get_gradient(&self, net: PropagatedNetwork, y: &Y, lambda: f32) -> Weights {
        let PropagatedNetwork {
            z,
            layers
        } = net;
        // Number of training examples
        let m = y.nrows() as f32;

        // Computed gradient
        let mut grad = vec![];

        // Partial derivative of the last layer
        let last_partial = &*layers[layers.len() - 1] - &**y;
        
        let mut ddelta: Weights = self.theta[..self.theta.len()].iter().map(|l| l.map(|_| 0.0)).collect();
        let range_m = m as usize;
    
        for i in 0..range_m {
            // Partial derivative of the last layer
            let d3 = last_partial.row(i).transpose();
            
            // All of the partials derivatives but last layer's
            let mut partials = vec![d3];
    
            // Gets a increasing index over the layers starting from 1
            for lx in 1..layers.len() {
                // Converts the increasing index to a decreasing index starting from the last
                let l = layers.len() - lx - 1;
    
                // Only computes the partial derivatives for the middle layers
                if l > 0 {
                    // Gets the z layer
                    let zr: DMatrix<f32> = z[l - 1].rows(i, 1).transpose();
                    
                    // Runs the gradient of the activation function(in this case sigmoid) over z layer
                    let sig = sigmoid_gradient( &zr ).insert_row(0, 1.0);
    
                    // Gets the partial derivative of the layer that comes after the current one
                    let next_partial = &partials[lx - 1];
    
                    let delta = (
                        // Multiplies the theta of the current layer with the partial derivative of the layer that comes after the current one
                        self.theta[l].transpose() * next_partial
                    ).component_mul(
                            // Multiplies the result with the activation function's gradient
                            &sig
                        );
                    // Saves the partial derivative of the current layer so it can be used to compute the theta of the layer that comes before the current one
                    partials.push(delta.clone().remove_row(0));
                }

                // Multiplies the partial derivative of the layer that comes after the current one with the values of the current layer
                ddelta[l] += &partials[lx - 1] * layers[l].row(i);            
            }
        }
    
        // Computes and caches first part of the regularization term
        let first_term = lambda / m;
    
        for (i, d) in ddelta.iter().enumerate() {
            grad.push(d / m); // Populates the grad with a not regularized gradient
            
            // Applies regularization for every theta but the bias one(0th)
            for (c, mut v) in grad[i].column_iter_mut().enumerate() {
                if c != 0 {
                    // Adds regularization term
                    v += first_term * self.theta[i].column(c);
                }
            }
        }

        return grad;
    }

    pub fn evaluate(&self, x: &X, y: &Y, lambda: f32) -> Evaluation {    
        let net = self.forward_propagation(&x);
    
        // It's not expansive to clone the arc references
        let cost = self.get_cost(net.layers.clone(), y, lambda);

        let grad = self.get_gradient(net, y, lambda);
    
        return Evaluation {
            cost,
            gradient: grad,
        };
    }

    pub fn take_step(&mut self, gradient: Weights, alpha: f32) {
        // Applies the gradient to theta
        for (i, grad) in gradient.iter().enumerate() {
            self.theta[i] -= grad * alpha;
        }
    }

    pub fn train(
        &mut self,
        x: &X,
        y: &Y,
        alpha: f32,
        lambda: f32,
        iters: usize,
        batch_size: usize,
        ctrlc_rx: Receiver<()>
    ) {
        if is_debugging() { println!("Training... "); }
        // Caches the mini batches so it won't have to clone it all over again
        let mut batches: Vec<[Arc<DMatrix<f32>>; 2]> = vec![];

        for i in 0..iters {
            // Checks for unexpected ctrlc signal
            if let Ok(_) = ctrlc_rx.try_recv() {
                break;
            }

            // Gets the mini batch index
            let v = i % (x.nrows() / batch_size);

            // Gets the mini batch ranges
            let v1 = v * batch_size;
            let v2 = v * batch_size + batch_size;

            // If the mini batch index is not on the cache then it clones and adds it to the cache
            if batches.len() <= v {
                let batch_x = x
                    .slice_range((v1)..(v2), ..).clone_owned();
                let batch_y = y
                    .slice_range((v1)..(v2), ..).clone_owned();

                batches.push([Arc::new(batch_x), Arc::new(batch_y)]);
            }
            
            // Gets the gradient and the cost for the current mini batch
            let evaluation = self.evaluate(&batches[v][0], &batches[v][1], lambda);

            // Applies the gradient to theta
            self.take_step(evaluation.gradient, alpha);
            
            if is_debugging() {
                print!("\rIteration: {i}, Mini batch: {v1}..{v2}, Cost: {}\r", evaluation.cost);
                io::stdout().flush().unwrap();
            }
        }
    }
}



