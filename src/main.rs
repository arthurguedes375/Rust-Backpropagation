extern crate nalgebra as na;

use na::{dmatrix, dvector};

pub mod sigmoid;
pub mod cost_function;

fn main() {

    let a = dmatrix![
        1.00000,   0.54030,  -0.41615;
        1.00000,  -0.98999,  -0.65364;
        1.00000,   0.28366,   0.96017;
    ];
    let y = dvector![
        4.0,
        2.0,
        3.0
    ];
    let weights = vec![
        dmatrix![
            0.10000,   0.30000,   0.50000;
            0.20000,   0.40000,   0.60000;
        ],
        dmatrix![
            0.70000,   1.10000,   1.50000;
            0.80000,   1.20000,   1.60000;
            0.90000,   1.30000,   1.70000;
            1.00000,   1.40000,   1.80000;
        ],
    ];

    let lambda = 4.0;

    let tst = cost_function::cost_function(
        a,
        weights,
        cost_function::Y::Indexed(y),
        lambda,
        false
    );

    println!("{}{:#?}", tst.0, tst.1)
}