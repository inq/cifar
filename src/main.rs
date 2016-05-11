extern crate monster;
extern crate rand;

mod cifar;

use monster::cudnn::{Cudnn, Tensor, Filter4d, Convolution2d, Pooling};
use monster::cudart::{Memory};
use monster::util::{Image};
use monster::{Nn};
use cifar::{Cifar};
use std::process::exit;
use std::env;
use rand::Rng;

fn layer(cudnn: &Cudnn,
         size: i32,
         chan_src: i32,
         chan_dst: i32,
         params: &Memory<f32>,
         bias: &Memory<f32>,
         ipt: &Memory<f32>)
         -> Result<Memory<f32>, &'static str> {
    let filter = try!(Filter4d::new(chan_dst, chan_src, 3, 3));
    let conv = try!(Convolution2d::new(1, 1, 1, 1, 1, 1));
    let src_tensor = try!(Tensor::new_4d(1, chan_src, size, size));
    let dst_tensor = try!(Tensor::new_4d(1, chan_dst, size, size));
    let bias_tensor = try!(Tensor::new_4d(1, chan_dst, 1, 1));
    let pool_tensor = try!(Tensor::new_4d(1, chan_dst, size / 2, size / 2));
    let mut mem = try!(Memory::<f32>::new((chan_dst * size * size) as usize));
    let res = try!(Memory::<f32>::new((chan_dst * size * size / 4) as usize));
    let pool = try!(Pooling::new_2d_max(2, 0, 2));

    try!(cudnn.conv_forward(&src_tensor,
                            &ipt,
                            &filter,
                            &params,
                            &conv,
                            &dst_tensor,
                            &mem));
    try!(cudnn.add_bias(&bias_tensor,
                        &bias,
                        &dst_tensor,
                        &mem));
    try!(cudnn.relu_forward_inplace(&dst_tensor,
                                    &mut mem));
    try!(cudnn.max_pooling_forward(&pool,
                                   &dst_tensor,
                                   &mem,
                                   &pool_tensor,
                                   &res));
    Ok(res)
}

fn run(args: Vec<String>) -> Result<(), &'static str> {
    let nn = try!(Nn::new());

    let cifar = try!(Cifar::new(&args[1]));
    let image = match cifar.images.iter().nth(9999) {
        Some(image) => image,
        _ => return Err("Could not read the image")
    };

    let mut rng = rand::thread_rng();

    // alloc device memory
    let src = try!(image.to_device());
    let params_conv1 = try!(Memory::<f32>::new(3 * 3 * 3 * 16));
    let mut tmp = vec![0f32; 3 * 3 * 3 * 16];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(params_conv1.write(&tmp));
    let bias_conv1 = try!(Memory::<f32>::new(16));
    let mut tmp = vec![0f32; 16];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(bias_conv1.write(&tmp));
        
    let data = try!(layer(&nn.cudnn,
                          32,
                          3,
                          16,
                          &params_conv1,
                          &bias_conv1,
                          &src));
    let params_conv2 = try!(Memory::<f32>::new(3 * 3 * 16 * 20));
    let mut tmp = [0.1f32; 3 * 3 * 16 * 20];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32) };
    try!(params_conv2.write(&tmp.to_vec()));
    let bias_conv2 = try!(Memory::<f32>::new(20));
    let mut tmp = vec![0f32; 20];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(bias_conv2.write(&tmp));
    
    let data2 = try!(layer(&nn.cudnn,
                           16,
                           16,
                           20,
                           &params_conv2,
                           &bias_conv2,
                           &data));
    let params_conv3 = try!(Memory::<f32>::new(3 * 3 * 20 * 20));
    let mut tmp = [0.1f32; 3 * 3 * 20 * 20];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32) };
    try!(params_conv3.write(&tmp.to_vec()));
    let bias_conv3 = try!(Memory::<f32>::new(20));
    let mut tmp = vec![0f32; 20];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(bias_conv3.write(&tmp));

    let data3 = try!(layer(&nn.cudnn,
                           8,
                           20,
                           20,
                           &params_conv3,
                           &bias_conv3,
                           &data2));
    // FCN
    let params_fcn = try!(Memory::<f32>::new(4 * 4 * 20 * 10));
    let mut tmp = vec![0.0f32; 4 * 4 * 20 * 10];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32) };
    try!(params_fcn.write(&tmp));
    let bias_fcn = try!(Memory::<f32>::new(10));
    let mut tmp = vec![0f32; 10];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(bias_fcn.write(&tmp));
    
    let mut data4_tensor = try!(Tensor::new_4d(1, 10, 1, 1));
    let mut bias_tensor = try!(Tensor::new_4d(1, 10, 1, 1));
    let mut data4 = try!(Memory::<f32>::new(10));
    try!(nn.fcn_forward(4 * 4 * 20, 10, &data3, &mut data4, &params_fcn));
    let mut tmp = vec![0f32; 10];
    try!(data4.read(&mut tmp));
    try!(nn.cudnn.add_bias(&bias_tensor,
                        &bias_fcn,
                        &data4_tensor,
                        &data4));
    println!("{:?}", tmp);

    // Softmax

    let mut data5_tensor = try!(Tensor::new_4d(1, 10, 1, 1));
    let mut data5 = try!(Memory::<f32>::new(10));
    try!(nn.cudnn.softmax_forward(&data4_tensor, &data4, &data5_tensor, &data5));
    let mut tmp = vec![0f32; 10];
    try!(data5.read(&mut tmp));
    println!("{:?}", tmp);

    // Loss Function - CrossEntropy
    let mut loss = - tmp[image.info as usize].ln();
    println!("loss: {}", loss);
    
    Ok(())
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
