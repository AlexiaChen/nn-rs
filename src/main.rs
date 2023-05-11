use ndarray::{Array, ArrayView, Dim};
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
        // mean 0.0 and standard deviation of 1 / sqrt(number of nodes of next layer) = inputnodes^(-0.5)
        let wih = Array::random((hiddennodes as usize, inputnodes as usize),
            Normal::new(0.0, (hiddennodes as f64).powf(-0.5)).unwrap());
        
        // outputnodes*hiddennodes matrix array
        // mean 0.0 and standard deviation of 1 / sqrt(number of nodes of next layer) = hiddennodes^(-0.5)
        let who = Array::random((outputnodes as usize, hiddennodes as usize),
            Normal::new(0.0, (outputnodes as f64).powf(-0.5)).unwrap());

       
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

    /// train the neural network
    fn train(&self) {
        println!("train");
    }

    /// query the neural network
    fn predict(&self, input_list: &Vec<f64>) -> Array<f64, Dim<[usize; 2]>>{
        if input_list.len() != self.input_nodes as usize {
            panic!("input list length does not match input nodes");
        }
        // convert input list to 2d array
        // calculate signals into hidden layer
        let input_vec = Array::from_shape_vec((input_list.len(), 1), input_list.clone()).unwrap();
        let hidden_input_vec = self.weight_ih.dot(&input_vec);
        // calculate the signals emerging from hidden layer
        let hidden_output_vec = hidden_input_vec.mapv(|x| (self.activation_function)(x));
        // calculate signals into final output layer
        let final_input_vec = self.weight_ho.dot(&hidden_output_vec);
        // calculate the signals emerging from final output layer
        let final_output_vec = final_input_vec.mapv(|x| (self.activation_function)(x));
        return final_output_vec;
    }
}


fn main() {
    let nn = NeuralNetwork::new(3, 3, 3, 0.3);
    println!("{:?}", nn);
    nn.train();
    //nn.predict();
}
