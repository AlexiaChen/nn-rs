
/// neural network  struct definition
#[derive(Debug)]
struct NeuralNetwork {
    input_nodes: i32,
    hidden_nodes: i32,
    output_nodes: i32,
    learning_rate: f64,
    wih: Vec<f64>, // weights from input to hidden layer
    who: Vec<f64>, // weights from hidden to output layer
}

impl NeuralNetwork {
    /// Create a new neural network from inputnodes, hiddennodes, outputnodes, learningrate
    fn new(inputnodes: i32, hiddennodes: i32, outputnodes: i32, learningrate: f64) -> NeuralNetwork {
        
        // link weight matrices, wih and who
        // weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        // w11 w21
        // w12 w22 etc 

        
        NeuralNetwork {
            // set number of nodes in each input, hidden, output layer
            input_nodes: inputnodes,
            hidden_nodes: hiddennodes,
            output_nodes: outputnodes,
            // learning rate
            learning_rate: learningrate,
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
