use crate::data::{load::load_csv as lcsv, Weights};

#[test]
fn load_csv() {
    let data = lcsv("src/tests/data/load_csv_test.csv", Some(0.8), false);

    assert_eq!(data.train.x.nrows(), 8);
    assert_eq!(data.test.x.nrows(), 2);

    assert_eq!(data.train.x.ncols(), 4);
    assert_eq!(data.test.x.ncols(), 4);

    assert_eq!(data.train.y.nrows(), 8);
    assert_eq!(data.test.y.nrows(), 2);
    
    assert_eq!(data.train.y.ncols(), 9);
    assert_eq!(data.test.y.ncols(), 10);
    
    assert_eq!(data.train.y.row(7).column(8)[0], 1.0);
    assert_eq!(data.test.y.row(0).column(9)[0], 1.0);
}

#[test]
fn init_theta() {
    let data = lcsv("src/tests/data/load_csv_test.csv", Some(0.8), false);
    let hidden_layers = 3;
    let hidden_layers_len = 30;
    let theta = Weights::init(
        &data.train.x,
        hidden_layers,
        hidden_layers_len,
        &data.train.y,
        None,
        false,
    );

    assert_eq!(theta[0].nrows(), hidden_layers_len as usize);
    assert_eq!(theta[0].ncols(), data.train.x.ncols());
    assert_eq!(theta.len() - 1, hidden_layers as usize);
    assert_eq!(theta[theta.len() - 1].nrows(), data.train.y.ncols());
    assert_eq!(theta[theta.len() - 1].ncols(), theta[theta.len() - 2].nrows() + 1);
}

#[test]
fn load_theta() {
    let theta = Weights::load("src/tests/data/theta_test.txt", false);

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