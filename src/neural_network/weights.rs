use rand::Rng;

use crate::{task::{is_debugging, init_task, end_task}, data::load::file_content};
use super::types::{LayerWeight};
use na::DMatrix;
use std::io::Write;

pub type Weights = Vec<LayerWeight>;

pub trait ManageWeights {
    fn init(
        input_nodes: u32,
        hidden_layers: u16,
        hidden_layers_len: u16,
        output_classes: u16,
        epsilon: Option<f32>
    ) -> Weights;
    fn info(&self);
    fn from_string(content: &str) -> Weights;
    fn from_file(filepath: &str) -> Weights;
    fn export(&self, folder_path: &str, filename_length: Option<usize>);
}

impl ManageWeights for Weights {
    fn init(
        input_nodes: u32,
        hidden_layers: u16,
        hidden_layers_len: u16,
        output_classes: u16,
        epsilon: Option<f32>
    ) -> Weights { 
        let epsilon = epsilon.unwrap_or(0.1);

        init_task("Initializing theta");
        let mut rand = rand::thread_rng();
        let theta_len = hidden_layers + 1;

        let mut weights: Weights = vec![];

        for i in 0..theta_len {
            let previous_len = if i == 0 {
                input_nodes as usize
            } else {
                hidden_layers_len as usize + 1
            };

            let next_len = if i == theta_len - 1 {
                output_classes as usize
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

    fn info(&self) {
        println!(
"
==Network Structure==
Hidden Layers: {},
Hidden layers len: {}
=====
", self.len() - 1, self[0].nrows() - 1,
        )
    }

    fn from_string(content: &str) -> Weights {
        init_task("Processing Weights");
        
        let mut weights = vec![];

        let layers = content.split('=').collect::<Vec<&str>>();

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

        if is_debugging() {
            weights.info();
        }

        return weights;
    }

    fn from_file(filepath: &str) -> Weights {
        init_task("Loading Weights");
        let content = file_content(filepath);
        end_task();
        

        return Weights::from_string(&content);
    }

    fn export(&self, folder_path: &str, filename_length: Option<usize>) {
            init_task("Exporting");
            let filename_length = filename_length.unwrap_or(9);
            let start = std::time::SystemTime::now();
            let mili = start
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_millis().to_string();
    
            let filepath = format!("{folder_path}/{}.txt", &mili[mili.len() - filename_length..]);
            let mut dump_file =  std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(filepath)
            .unwrap();
    
            for (i, layer) in (&self).iter().enumerate() {
                for row in layer.row_iter() {
                    for col in row.column_iter() {
                        write!(dump_file, "{},", col[0]).unwrap();
                    }
                    write!(dump_file, "\n").unwrap();
                }
                if i < self.len() - 1 {
                    write!(dump_file, "=").unwrap();
                }
            }
    
            end_task();
    }
}