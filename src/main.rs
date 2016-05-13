extern crate monster;
extern crate rand;

mod cifar;

use monster::cudnn::{Cudnn, Tensor, Pooling};
use monster::cudart::{Memory};
use monster::util::{Image};
use monster::{Nn};
use cifar::{Cifar};
use std::process::exit;
use std::env;
use rand::Rng;

struct LayerData {
    after_conv: Tensor,
    after_relu: Tensor,
    after_pool: Tensor
}

struct LayerParam {
    conv: Tensor,
    bias: Tensor
}

fn layer(nn: &Nn,
         ipt: &Tensor,
         size: i32,
         chan_src: i32,
         chan_dst: i32,
         params: &LayerParam)
         -> Result<LayerData, &'static str> {
    let filter = try!(Nn::new_filter(chan_dst, chan_src, 3, 3));
    let conv = try!(Nn::new_conv(1, 1, 1));
    let pool = try!(Nn::new_max_pooling(2, 0, 2));
    let data = LayerData {
        after_conv: try!(Tensor::new(1, chan_dst, size, size)),
        after_relu: try!(Tensor::new(1, chan_dst, size, size)),
        after_pool: try!(Tensor::new(1, chan_dst, size / 2, size / 2))
    };
    try!(nn.conv_forward(ipt, &filter, &params.conv, &conv, &data.after_conv));
    try!(nn.bias_forward(&data.after_conv, &params.bias));
    try!(nn.relu_forward(&data.after_conv, &data.after_relu));
    try!(nn.pooling_forward(&pool, &data.after_relu, &data.after_pool));
    Ok(data)
}

fn run(args: Vec<String>) -> Result<(), &'static str> {
    let nn = try!(Nn::new());

    let cifar = try!(Cifar::new(&args[1]));
    let image = match cifar.images.iter().nth(9999) {
        Some(image) => image,
        _ => return Err("Could not read the image")
    };

    let mut rng = rand::thread_rng();

    // Initialize parameters
    let layers = [
        LayerParam {
            conv: try!(Tensor::new(1, 3 * 16, 3, 3)),
            bias: try!(Tensor::new(1, 16, 1, 1))
        },
        LayerParam {
            conv: try!(Tensor::new(1, 16 * 20, 3, 3)),
            bias: try!(Tensor::new(1, 20, 1, 1))
        },
        LayerParam {
            conv: try!(Tensor::new(1, 20 * 20, 3, 3)),
            bias: try!(Tensor::new(1, 20, 1, 1))
        }
    ];
    let mut tmp = vec![0f32; 3 * 3 * 3 * 16];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(layers[0].conv.write(&tmp));
    let mut tmp = vec![0f32; 16];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(layers[0].bias.write(&tmp));

    let mut tmp = [0.1f32; 3 * 3 * 16 * 20];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32) };
    try!(layers[1].conv.write(&tmp.to_vec()));
    let mut tmp = vec![0f32; 20];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(layers[1].bias.write(&tmp));

    let mut tmp = [0.1f32; 3 * 3 * 20 * 20];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32) };
    try!(layers[2].conv.write(&tmp.to_vec()));
    let mut tmp = vec![0f32; 20];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(layers[2].bias.write(&tmp));

    let params_fcn = try!(Tensor::new(1, 4 * 4 * 20 * 10, 1, 1));
    let mut tmp = vec![0.0f32; 4 * 4 * 20 * 10];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32) };
    try!(params_fcn.write(&tmp));
    let bias_fcn = try!(Tensor::new(1, 10, 1, 1));
    let mut tmp = vec![0f32; 10];
    for i in 0..tmp.len() { tmp[i] = rng.gen_range(-0.1f32, 0.1f32); }
    try!(bias_fcn.write(&tmp));
    // alloc device memory
    for i in 0..10 {
        let src = try!(image.to_device());
        let data = try!(layer(&nn,
                              &src,
                              32,
                              3,
                              16,
                              &layers[0]));
        let data2 = try!(layer(&nn,
                               &data.after_pool,
                               16,
                               16,
                               20,
                               &layers[1]));
        let data3 = try!(layer(&nn,
                               &data2.after_pool,
                               8,
                               20,
                               20,
                               &layers[2]));
        // FCN
        let mut data4 = try!(Tensor::new(1, 10, 1, 1));
        try!(nn.fcn_forward(&data3.after_pool, &data4, &params_fcn));
        let mut tmp = vec![0f32; 10];
        try!(data4.read(&mut tmp));
        try!(nn.bias_forward(&data4,
                             &bias_fcn));
        // Softmax
        let mut data5 = try!(Tensor::new(1, 10, 1, 1));
        try!(nn.softmax_forward(&data4, &data5));
        let mut tmp = vec![0f32; 10];
        try!(data5.read(&mut tmp));

        // Loss Function - CrossEntropy
        let mut loss = - tmp[image.info as usize].ln();
        println!("loss: {}", loss);

        // Back Propagation!
        let scale = 0.01f32;

        let mut dy = try!(Tensor::new(1, 10, 1, 1));
        let mut target = vec![0f32; 10];
        target[image.info as usize] = 1f32;
        try!(dy.write(&target));
        for i in 0..target.len() { target[i] = target[i] - tmp[i] };

        // Softmax
        let mut dx = try!(Tensor::new(1, 10, 1, 1));
        try!(nn.softmax_backward(&data5,
                                 &dy,
                                 &dx));
        let mut tmp = vec![0f32; 10];
        dx.read(&mut tmp);

        let mut dy = dx;
        let mut dx = try!(Tensor::new(1, 20, 4, 4));

        // FCN
        try!(nn.bias_backward(scale,
                              &bias_fcn,
                              &dy));

        try!(nn.fcn_backward(scale,
                             &data3.after_pool,
                             &dy,
                             &dx,
                             &params_fcn));
    }    
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
