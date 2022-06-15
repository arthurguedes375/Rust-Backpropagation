use crate::data::{X, Y};
use csv::{Reader, StringRecord};
use crate::task::{init_task, end_task};

pub struct Data {
    pub x: X,
    pub y: Y,
}

pub struct Dataset {
    pub train: Data,
    pub test: Data,
}

pub fn load_csv(path: &str, train_percentage: Option<f32>) -> Dataset {
    init_task("Loading csv");

    let mut content = Reader::from_path(path).unwrap(); 
    let value = content.records().map(|x| x.unwrap()).collect::<Vec<StringRecord>>();
    let m = value.len();
    let a1_len = value[0].len();

    end_task();
    init_task("Processing csv");

    let mut x = na::DMatrix::<f32>::zeros(
        m,
        a1_len,
    );

    let mut y = na::DVector::<f32>::zeros(m);


    for (i, record) in value.iter().enumerate() {
        y[i] = record[0].parse().unwrap();
        for (ip, pixels) in record.iter().enumerate() {
            if ip == 0 {
                continue;
            }

            x.row_mut(i).column_mut(ip - 1)[0] = pixels.parse().unwrap();
        }
    }

    let train_percentage = train_percentage.or(Some(0.60)).unwrap();

    let divisor = (m as f32 * train_percentage) as usize;

    let x_train = x.slice_range(..divisor, ..).clone_owned();
    let y_train = y.slice_range(..divisor, ..).clone_owned();

    let x_test = x.slice_range(divisor.., ..).clone_owned();
    let y_test = y.slice_range(divisor.., ..).clone_owned();

    end_task();
    return Dataset {
        train: Data {
            x: X::NotBiased(x_train),
            y: Y::Indexed(y_train),
        },
        test: Data {
            x: X::NotBiased(x_test),
            y: Y::Indexed(y_test),
        },
    };
}