use na::{dmatrix, dvector};

use crate::data::{load::Dataset, X, Y, ManageX, ManageY};

#[test]
fn load_csv() {
    let data = Dataset::from_csv_file("src/tests/data/load_csv_test.csv", Some(0.8));

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
fn to_biased() {
    let data = X::from_unbiased(dmatrix![2.0, 3.0, 4.0; 5.0, 6.0, 7.0]);

    assert_eq!(data.column(0)[0], 1.0);
    assert_eq!(data.column(0)[1], 1.0);
}

#[test]
fn to_y_matrix() {
    let m = Y::from_indexed_y(dvector![1.0, 2.0, 3.0, 0.0]);

    assert_eq!(m.row(0).column(1)[0], 1.0);
    assert_eq!(m.row(1).column(2)[0], 1.0);
    assert_eq!(m.row(2).column(3)[0], 1.0);
    assert_eq!(m.row(3).column(0)[0], 1.0);
}