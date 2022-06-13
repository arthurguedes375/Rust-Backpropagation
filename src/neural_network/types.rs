use na::{DVector, DMatrix};

pub type Unit = f32;
pub type Weight = f32;

pub type Layer = DVector<Unit>;
pub type LayerWeight = DMatrix<Weight>; //actual layer X layer_before + bias

pub type Layers = Vec<Layer>;
pub type Weights = Vec<LayerWeight>;