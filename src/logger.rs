use std::fmt::Debug;
use std::fs::File;
use std::io::Write;
pub struct Logger<T: Write> {
    stream: T,
}

impl<T> Logger<T>
where
    T: Write,
{
    pub fn new(output_stream: T) -> Logger<T> {
        Logger {
            stream: output_stream,
        }
    }

    pub fn log(&mut self, info: &str) {
        self.stream.write_all(info.as_bytes()).unwrap();
        self.stream.flush().unwrap();
    }

    pub fn dump(&mut self, name: Option<&str>, data: &impl Debug) {
        let buffer = format!("{:#?}", data);
        if let Some(name) = name {
            let name = name.to_owned() + ": ";
            self.stream.write_all(name.as_bytes()).unwrap();
        }
        self.stream.write_all(buffer.as_bytes()).unwrap();
        self.stream.flush().unwrap();
    }
}

pub fn create_logfile() -> impl Write {
    let logfile = File::options()
        .write(true)
        .create(true)
        .truncate(true)
        .open("logfile2.txt")
        .expect("cant open logfile for writing.");
    logfile
    // BufWriter::new(logfile)
}
