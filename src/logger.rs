use std::fmt::Debug;
use std::fs::File;
use std::io::Write;

pub trait Logger {
    fn log(&mut self, info: &str);
    fn dump(&mut self, name: Option<&str>, data: Box<dyn Debug>);
}

pub struct FileLogger {
    stream: File,
}

impl FileLogger {
    pub fn new(output_stream: File) -> Self {
        FileLogger {
            stream: output_stream,
        }
    }
}

impl Logger for FileLogger {
    fn log(&mut self, info: &str) {
        self.stream.write_all(info.as_bytes()).unwrap();
        self.stream.flush().unwrap();
    }

    fn dump(&mut self, name: Option<&str>, data: Box<dyn Debug>) {
        let buffer = format!("{:#?}", data);
        if let Some(name) = name {
            let name = name.to_owned() + ": ";
            self.stream.write_all(name.as_bytes()).unwrap();
        }
        self.stream.write_all(buffer.as_bytes()).unwrap();
        self.stream.flush().unwrap();
    }
}

pub fn create_logfile() -> File {
    let logfile = File::options()
        .write(true)
        .create(true)
        .truncate(true)
        .open("logfile2.txt")
        .expect("cant open logfile for writing.");
    logfile
    // BufWriter::new(logfile)
}
