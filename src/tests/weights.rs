use crate::data::{ManageX, ManageY};

use super::super::{
    data::load::Dataset,
    neural_network::weights::{ManageWeights, Weights},
};

#[test]
fn init_theta() {
    let data = Dataset::from_csv_file("src/tests/data/load_csv_test.csv", Some(0.8));
    let hidden_layers = 3;
    let hidden_layers_len = 30;
    let theta = Weights::init(
        data.train.x.input_nodes(),
        hidden_layers,
        hidden_layers_len,
        data.train.y.output_classes(),
        None,
    );

    assert_eq!(theta[0].nrows(), hidden_layers_len as usize);
    assert_eq!(theta[0].ncols(), data.train.x.ncols());
    assert_eq!(theta.len() - 1, hidden_layers as usize);
    assert_eq!(theta[theta.len() - 1].nrows(), data.train.y.ncols());
    assert_eq!(theta[theta.len() - 1].ncols(), theta[theta.len() - 2].nrows() + 1);
}

#[test]
fn load_theta() {
    let theta = Weights::from_file("src/tests/data/theta_test.txt");

    /*
     These values come from the theta_test exported theta,
     don't change this unless you changed the file(re-export another set of theta)
    */
    let hidden_layers = 2;
    let hidden_layers_len = 16;
    let xcols = 786;
    let ycols = 10;
    // ===========================================

    assert_eq!(theta[0].nrows(), hidden_layers_len);
    assert_eq!(theta[0].ncols(), xcols);
    assert_eq!(theta.len() - 1, hidden_layers);
    assert_eq!(theta[theta.len() - 1].nrows(), ycols);
    assert_eq!(theta[theta.len() - 1].ncols(), theta[theta.len() - 2].nrows() + 1);
}