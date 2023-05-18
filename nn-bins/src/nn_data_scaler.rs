use ndarray::{Array, Dim};
use std::fs::File;
use std::io::prelude::*;
use csv::{ReaderBuilder, StringRecord};
use std::error::Error;


fn get_img_pixels_data(record: &StringRecord) -> Vec<u8> {
    let mut image_vec: Vec<u8> = vec![];
    let mut count = 0;
    for data_value in record.iter() {
        if count == 0 {
            count += 1;
            continue;
        } else {
            image_vec.push(data_value.parse::<u8>().unwrap());
        }
    }
    image_vec
}

fn main() -> Result<(), Box<dyn Error>> {
   
    let mut file = File::open("./dataset/mnist_train/file0.csv").expect("you must download the dataset first, https://pjreddie.com/projects/mnist-in-csv/");
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_reader(contents.as_bytes());

    let mut count = 0;
    let mut records: Vec<StringRecord> = vec![];
    for result in reader.records() {
        if count == 100 {
            break;
        }
        let record = result?;
        records.push(record);
        count += 1;
    }

    println!("Records len: {:?}", records.len());
    
    let record_0 = &records[0];
    let image_vec: Vec<u8> = get_img_pixels_data(record_0);
    let image_vec: Vec<f32> = image_vec.iter().map(|x| *x as f32).collect();

    
    let image_array: Array<f32, Dim<[usize; 2]>> = Array::from_shape_vec((28, 28), image_vec).unwrap();

    let  scaled_data = image_array / 255.0 * 0.99 + 0.01;
    println!("Scaled data: {:?}", scaled_data);

    Ok(())
   
}