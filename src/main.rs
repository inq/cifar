extern crate monster;

use monster::cudnn::{Cudnn, Tensor, Filter4d, Convolution2d};
use monster::cudart::{Memory};
use monster::util::{Image};
use monster::cifar::{Cifar};
use std::process::exit;
use std::path::Path;
use std::env;

fn run(args: Vec<String>) -> Result<(), &'static str> {
    let cudnn = try!(Cudnn::new());

    let cifar = try!(Cifar::new(&args[1]));
    let image = match cifar.images.iter().nth(9999) {
        Some(image) => image,
        _ => return Err("Could not read the image")
    };

    // alloc device memory
    let filter = try!(Filter4d::new(3, 3, 3, 3));
    let conv = try!(Convolution2d::new(1, 1, 1, 1, 1, 1));
    let src_tensor = try!(Tensor::new_4d(1, 3, 32, 32));
    let dst_tensor = try!(Tensor::new_4d(1, 3, 32, 32));
    let (n, c, h, w) = try!(conv.get_forward_output_dim(&src_tensor, &filter));
    let mut vals = try!(Memory::<f32>::new(10 * 10 * 10));
    let tmp = [0.1f32; 10 * 10 * 10];
    try!(vals.write(&tmp.to_vec()));
    
    let mut dst = try!(Memory::<f32>::new(32 * 32 * 3));
    let src = try!(image.to_device());
    try!(cudnn.conv_forward(&src_tensor,
                            &src,
                            &filter,
                            &vals,
                            &conv,
                            &dst_tensor,
                            &mut dst));
    let img = try!(Image::from_device(dst, 1u8, 32, 32));

    // write png image
    img.save("images/cifar.png")
}

fn main() {
    match run(env::args().collect()) {
        Ok(_) => exit(0),
        Err(e) => {
            println!("{}", e);
            exit(1)
        }
    }
}
