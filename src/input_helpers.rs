use std::{io, io::Write, str::FromStr};

pub fn ask<T>(question: &str) -> Result<T, T::Err>
where
    T: FromStr,
{
    print!("{}: ", question);
    io::stdout().flush().unwrap();
    let mut buff = String::new();
    io::stdin()
        .read_line(&mut buff)
        .expect("cant read from stdin");
    buff.trim().parse()
}
