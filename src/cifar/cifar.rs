use std::collections::LinkedList;
use std::fs::File;
use std::path::Path;
use std::io::Read;

use monster::util::Image;

#[allow(dead_code)]
pub struct Cifar {
    path: String,
    pub images: LinkedList<Image<u8>>
}

impl Cifar {
    fn read_img(file: &mut File) -> Result<Image<u8>, &'static str> {
        let mut buf : [u8; 1] = [0];

        try!(match file.read(&mut buf) {
            Ok(len) if len == buf.len() => Ok(()),
            _ => Err("EOF")
        });
        Image::from_file(file, 32, 32, buf[0])
    }
    
    pub fn new(loc: &String) -> Result<Cifar, &'static str> {
        let path = Path::new(loc).join("data_batch_1.bin");
        let mut file = match File::open(path) {
            Ok(file) => file,
            _ => return Err("Could not read the data file")
        };
        let mut imgs = LinkedList::new();
        loop {
            match Cifar::read_img(&mut file) {
                Ok(img) => imgs.push_back(img),
                _ => break
            }
        }
        Ok(Cifar { path: loc.clone(), images: imgs })
    }
}
