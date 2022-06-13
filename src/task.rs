use std::io::Write;

pub fn init_task(message: &str) {
    print!("{}... ", message);
    std::io::stdout().flush().unwrap();
}

pub fn end_task() {
    println!("Ok.")
}