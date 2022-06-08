use nalgebra::{DMatrix};
use std::{f32::consts::E};

pub fn sigmoid(value: &DMatrix<f32>) -> DMatrix<f32>
{
    value.map(|v| {
        1.0 / (1.0 + (1.0 / E.powf(v)))
    })
}

pub fn sigmoid_gradient(value: &DMatrix<f32>) -> DMatrix<f32> {
    let sig = sigmoid(&value);
    sig.component_mul(&sig.map(|x| 1.0 - x))
}