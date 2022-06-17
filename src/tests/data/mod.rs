use crate::data::load::load_csv as lcsv;
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