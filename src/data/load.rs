use crate::{data::{X, Y, ManageX, ManageY}, task::is_debugging};
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

impl Dataset {
    pub fn from_csv_string(content: &str, train_percentage: Option<f32>) -> Dataset {
        init_task("Processing csv");

        let mut content = Reader::from_reader(content.as_bytes()); 
        let value = content.records().map(|x| x.unwrap()).collect::<Vec<StringRecord>>();
        let m = value.len();
        let a1_len = value[0].len();
    
        
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
    
        let x_train: X = X::from_unbiased(x.slice_range(..divisor, ..).clone_owned());
        let y_train: Y = Y::from_indexed_y(y.slice_range(..divisor, ..).clone_owned());
    
        let x_test: X = X::from_unbiased(x.slice_range(divisor.., ..).clone_owned());
        let y_test: Y = Y::from_indexed_y(y.slice_range(divisor.., ..).clone_owned());
    
        end_task();
    
        if is_debugging() {
                println!(
    "========
Training examples: {}
Output layers: {}
========
    ", x.nrows(), y.ncols()
                );
            }
        return Dataset {
            train: Data {
                x: x_train,
                y: y_train,
            },
            test: Data {
                x: x_test,
                y: y_test,
            },
        };
    }

    pub fn from_csv_file(path: &str, train_percentage: Option<f32>) -> Dataset {
        init_task("Loading csv file");
        let content = file_content(path);
        end_task();
        Dataset::from_csv_string(&content, train_percentage)
    }
}

pub fn file_content(path: &str) -> String {
    std::fs::read_to_string(path).unwrap()
}