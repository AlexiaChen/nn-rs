
/// neural network  struct definition
#[derive(Debug)]
struct NeuralNetwork {
    input_nodes: i32,
    hidden_nodes: i32,
    output_nodes: i32,
    learning_rate: f64,
}

impl NeuralNetwork {
    /// Create a new neural network from inputnodes, hiddennodes, outputnodes, learningrate
    fn new(inputnodes: i32, hiddennodes: i32, outputnodes: i32, learningrate: f64) -> NeuralNetwork {
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
