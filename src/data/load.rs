use crate::data::{X, Y};
use csv::{Reader, StringRecord};
use crate::task::{init_task, end_task};

pub struct Data {
    pub x: X,
    pub y: Y,
}

pub fn load_csv(path: &str) -> Data {
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

    end_task();
    return Data {
        x: X::NotBiased(x),
        y: Y::Indexed(y),
    };
}