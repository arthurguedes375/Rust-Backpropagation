use std::io::Write;

pub fn is_debugging() -> bool {
    match std::env::var("NEURAL_NETWORK_DEBUG").unwrap_or("0".into()).parse::<u8>().unwrap() {
        1 => true,
        _ => false,
    }
}

pub fn init_task(message: &str) {
    if !is_debugging() { return; }
    print!("{}... ", message);
    std::io::stdout().flush().unwrap();
}

pub fn end_task() {
    if !is_debugging() { return; }
    println!("Ok.")
}