use std::thread;
use std::sync::{mpsc::{channel, Receiver, Sender}, Mutex, Arc};

pub mod types;

use std::io::{self, Write};
use std::time::Duration;
use std::{fs, time};

use crate::task::{init_task, end_task};
use na::{DMatrix};
use types::{Weights, Unit};
use crate::data::{X, Y};

use crate::sigmoid::{sigmoid, sigmoid_gradient};

pub struct ToWorker {
    last_partial: DMatrix<f32>,
    layers: Vec<DMatrix<f32>>,
    z: Vec<DMatrix<f32>>,
    theta: Weights,
    ddelta: Arc<Mutex<Weights>>,
}

pub type FromWorker = ();
pub type Workers = Vec<Sender<ToWorker>>;

pub struct PropagatedNetwork {
    pub z: Vec<DMatrix<Unit>>,
    pub layers: Vec<DMatrix<Unit>>,
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
        let x = x.val();
        let mut layers = vec![x.clone()];
        let mut zs = vec![];
    
        for (i, weights) in self.theta.iter().enumerate() {
            let z = &layers[i] * weights.transpose();
            zs.push(z.clone());
            let sig = sigmoid(&z);
            let new_layer = if i == self.theta.len() - 1 {
                sig
            } else {
                sig.insert_column(0, 1.0)
            };
            layers.push(new_layer);
        }
    
        return PropagatedNetwork {
            z: zs,
            layers,
        };
    }

    pub fn precision(&self, x: &X, y: &Y) -> f32 {
        init_task("Precision: Forward Prop");
        let results = self.forward_propagation(x);
        let y = y.val();
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

    pub fn cost_function(
        &self,
        x: &X,
        y: &Y,
        lambda: f32,
        threads: Option<u8>,
        workers_opt: Option<(Workers, &Receiver<FromWorker>)>
    ) -> Evaluation {
        let threads = threads.unwrap_or(1);
        let wo ;
        let (workers, result_rx);

        if let Some(w) = workers_opt {
            workers = w.0;
            result_rx = w.1;
        } else {
            wo = NeuralNetwork::init_workers(threads);
            workers = wo.0;
            result_rx = &wo.1;
        }

        let x = x.val();
        let y = y.val();
        let m = x.row_iter().len() as f32;
        let mut grad = vec![];
    
        let PropagatedNetwork {z, layers} = self.forward_propagation(&X::Biased(x));
    
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
    
        let last_partial = &layers[layers.len() - 1] - y;

        let ddelta: Arc<Mutex<Weights>> = Arc::new(
        Mutex::new(
                self.theta[..self.theta.len()]
                    .iter()
                    .map(|l| {
                        DMatrix::<f32>::zeros(l.nrows(), l.ncols())
                    })
                    .collect()
                )
            );
        let range_m = m as usize;

        let batch_size = range_m / threads as usize;

        for (i, worker) in workers.iter().enumerate() {
            let v1 = i * batch_size;
            let v2 = v1 + batch_size;

            let batch_last_partial = last_partial.slice_range(v1..v2, ..).clone_owned();
            let batch_layers = layers.iter().map(|layer| {
                layer.slice_range(v1..v2, ..).clone_owned()
            }).collect();
            let batch_z = z.iter().map(|z| {
                z.slice_range(v1..v2, ..).clone_owned()
            }).collect();

            worker.send(ToWorker {
                last_partial: batch_last_partial,
                layers: batch_layers,
                z: batch_z,
                theta: self.theta.clone(),
                ddelta: ddelta.clone()
            }).unwrap();
        }

        let mut finished = 0;

        while finished < workers.len() {
            result_rx.recv().unwrap();
            finished += 1;
        }
    
        let first_term = lambda / m;
    
        let ddelta_lock = ddelta.lock().unwrap();
        for (i, d) in ddelta_lock.iter().enumerate() {
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
    
    pub fn worker(data: Receiver<ToWorker>, from_worker: Sender<FromWorker>)  {
        for ToWorker {
            last_partial,
            layers,
            z,
            theta,
            ddelta,
        } in data {
            let m = last_partial.nrows();
            for i in 0..m {
                let d3 = last_partial.row(i).transpose();
                
                let mut partials = vec![d3];
        
                for lx in 1..layers.len() {
                    let l = layers.len() - lx - 1;
        
                    if l > 0 {
                        let zr: DMatrix<f32> = z[l - 1].rows(i, 1).transpose(); // Z
                        
                        let sig = sigmoid_gradient( &zr ).insert_row(0, 1.0);
        
                        let next_partial = &partials[lx - 1];
        
                        let delta = (
                            &theta[l].transpose() * next_partial
                        ).component_mul(
                                &sig
                            );
                        partials.push(delta.clone().remove_row(0));
                    }
                    
                    let mut ddelta_lock = ddelta.lock().unwrap();
                    (*ddelta_lock)[l] += &partials[lx - 1] * layers[l].row(i);
                    drop(ddelta_lock); // Unlocks ddelta
                }
                thread::sleep(Duration::from_nanos(1));
            }

            from_worker.send(()).unwrap();
        }
    }

    pub fn init_workers(threads: u8) -> (Workers, Receiver<FromWorker>) {
        let (parent_tx, result_rx) = channel::<FromWorker>();
        let core_ids = core_affinity::get_core_ids().unwrap();

        let mut workers = vec![];
        for i in 0..threads {
            let (to_worker, from_worker) = channel::<ToWorker>();
            workers.push(to_worker);

            let pr = parent_tx.clone();
            let id = core_ids[i as usize %core_ids.len()];

            thread::spawn(move || {
                core_affinity::set_for_current(id);
                NeuralNetwork::worker(from_worker, pr);
            });
        }

        return (workers, result_rx);
    }
    
    pub fn train(
        &mut self,
        x: &X,
        y: &Y,
        alpha: f32,
        lambda: f32,
        iters: usize,
        batch_size: usize,
        threads: Option<u8>,
        debug: bool,
        ctrlc_rx: Receiver<()>
    ) {
        init_task("Initializing Workers... ");
        let threads = threads.unwrap_or(1);
        let (workers, result_rx) = NeuralNetwork::init_workers(threads);
        end_task();


        for i in 0..iters {
            if let Ok(_) = ctrlc_rx.try_recv() {
                break;
            }
            
            let x = x.val();
            let y = y.val();
            let v = i % (x.nrows() / batch_size);
            
            let v1 = v * batch_size;
            let v2 = v1 + batch_size;

            let batch_x = x
                .slice_range((v1)..(v2), ..).clone_owned();
            let batch_y = y
                .slice_range((v1)..(v2), ..).clone_owned();

            let evaluation = self.cost_function(
                &X::Biased(batch_x),
                &Y::Matrix(batch_y),
                lambda,
                Some(threads),
                Some((workers.clone(), &result_rx))
            );

            for (i, grad) in evaluation.gradient.iter().enumerate() {
                self.theta[i] -= grad * alpha;
            }

            if debug {
                print!("\rIteration: {i}, Workers: {}, Batch: {v1}..{v2}, Cost: {}\r", workers.len(), evaluation.cost);
                io::stdout().flush().unwrap();
            }
        }
    }

    pub fn export(&self, path: &str) {
        init_task("Exporting");
        let start = time::SystemTime::now();
        let mili = start
            .duration_since(time::UNIX_EPOCH)
            .unwrap().as_millis();

        let filepath = format!("{path}/{}.txt", mili);
        let mut dump_file =  fs::OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open(filepath)
        .unwrap();

        for (i, layer) in (&self.theta).iter().enumerate() {
            for row in layer.row_iter() {
                for col in row.column_iter() {
                    write!(dump_file, "{},", col[0]).unwrap();
                }
                write!(dump_file, "\n").unwrap();
            }
            if i < self.theta.len() - 1 {
                write!(dump_file, "=").unwrap();
            }
        }

        end_task()
    }
}



