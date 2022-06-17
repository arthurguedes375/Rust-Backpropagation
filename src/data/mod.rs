pub mod load;

use std::sync::Arc;

use na::{DVector, DMatrix};
use crate::neural_network::types::{Unit};

pub type X = Arc<DMatrix<Unit>>;
pub type Y = Arc<DMatrix<f32>>; // m X k

pub trait ManageX {
    fn from_unbiased(v: DMatrix<Unit>) -> X;
    fn input_nodes(&self) -> u32;
}

pub trait ManageY {
    fn from_indexed_y(v: DVector<Unit>) -> Y;
    fn output_classes(&self) -> u16;
}

impl ManageX for X {
    fn from_unbiased(v: DMatrix<Unit>) -> X {
        Arc::new(v.insert_column(0, 1.0))
    }

    fn input_nodes(&self) -> u32 {
        self.ncols() as u32
    }
}

impl ManageY for Y {
    fn from_indexed_y(v: DVector<Unit>) -> Y {
        let m = v.row_iter().len();
        let mut result = DMatrix::<f32>::zeros(m, v.max() as usize + 1);
        for (row, index) in v.iter().enumerate() {
            result.row_mut(row).column_mut(*index as usize)[0] = 1.0; 
        }
    
        Arc::new(result)
    }

    fn output_classes(&self) -> u16 {
        self.ncols() as u16
    }
}