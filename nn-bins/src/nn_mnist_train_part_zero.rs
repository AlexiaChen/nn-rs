use nn::NeuralNetwork;

fn main() {
    // Why hidden nodes is 100?
    // Because there is no scientific way to determine the number of hidden nodes, we think neural network should find some patterns in the input data, 
    // these patterns can be represented by the hidden nodes with shorter length., so we did not choose number which is larger than 28*28. That can force
    // neural network to find some patterns in the input data. But if you choose a number which is too small, neural network will not find some patterns
    // you must konw that there is no best way to determine the number of hidden nodes. The better way is to try different numbers and find the best one.

    // Why output nodes is 10?
    // Because there are 10 digits(0,1,2,3,4,5,6,7,8,9) in the MNIST dataset, so we choose 10 as the output nodes.
    let mut nn = NeuralNetwork::new(28 * 28, 100, 10, 0.3);
    println!("NN is: {:?}", nn);
    let input_list = vec![1.0, 0.5, -1.5];
    let target_list = vec![0.5, 1.0, 0.5];
    let o = nn.predict(&input_list);
    println!("Output Vector Before train is: {:?}", o);
    nn.train(&input_list, &target_list);
    let o = nn.predict(&input_list);
    println!("Output Vector After train is: {:?}", o);
   
}