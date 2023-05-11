use ndarray::{Array, Dim};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

/// neural network  struct definition
#[derive(Debug)]
struct NeuralNetwork {
    input_nodes: i32,
    hidden_nodes: i32,
    output_nodes: i32,
    learning_rate: f64,
    weight_ih: Array<f64, Dim<[usize; 2]>>, // weights matrix from input to hidden layer
    weight_ho: Array<f64, Dim<[usize; 2]>>, // weights matrix from hidden to output layer
    activation_function: fn(f64) -> f64,

}

impl NeuralNetwork {
    /// Create a new neural network from inputnodes, hiddennodes, outputnodes, learningrate
    fn new(inputnodes: i32, hiddennodes: i32, outputnodes: i32, learningrate: f64) -> NeuralNetwork {
        
        // link weight matrices, wih and who
        // weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        // w11 w21
        // w12 w22 etc 

        // hiddennodes*inputnodes matrix array
        // mean 0.0 and standard deviation of 1 / sqrt(number of incoming links) = inputnodes^(-0.5)
        let wih = Array::random((hiddennodes as usize, inputnodes as usize),
            Normal::new(0.0, (inputnodes as f64).powf(-0.5)).unwrap());
        
        // outputnodes*hiddennodes matrix array
        // mean 0.0 and standard deviation of 1 / sqrt(number of hidden links) = hiddennodes^(-0.5)
        let who = Array::random((outputnodes as usize, hiddennodes as usize),
            Normal::new(0.0, (hiddennodes as f64).powf(-0.5)).unwrap());

       
        // 1 / (1 + e^(-x))
        fn sigmoid(x: f64) -> f64 {
            1.0 / (1.0 + (-x).exp())
        }
        
        NeuralNetwork {
            // set number of nodes in each input, hidden, output layer
            input_nodes: inputnodes,
            hidden_nodes: hiddennodes,
            output_nodes: outputnodes,
            weight_ih: wih,
            weight_ho: who,
            // learning rate
            learning_rate: learningrate,
            // activation function is the sigmoid function
            activation_function: |x| sigmoid(x),
        }
    }

    fn train(&self) {
        println!("train");
    }

    fn predict(&self) {
        println!("predict");
    }
}


fn main() {
    let nn = NeuralNetwork::new(3, 3, 3, 0.3);
    println!("{:?}", nn);
    nn.train();
    nn.predict();
}
