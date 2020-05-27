
use std::collections::HashMap;
use rand::Rng;

#[derive(Debug)]
struct Nexus {
    forward: f32,
    backward: f32,
}

impl Nexus {
    fn new() -> Nexus {
        let mut rng = rand::thread_rng();
        let mut forward = -0.01f32 + 0.02f32*rng.gen::<f32>();
        let mut backward = -0.01f32 + 0.02f32*rng.gen::<f32>();
        Nexus{forward:forward, backward:backward}
    }
}

struct Encoder<'a> {
    vec_size: usize,                        // dimensionality of word embeddings
    epochs: usize,                          // # of training ephocs completed
    vocab: HashMap<&'a str, u32>,              // frequency of vocab occurence
    output: HashMap<&'a str, u32>,
    weights: HashMap<(&'a str, usize), Nexus>, // weights
    hidden: HashMap<usize, f32>,
    errors: HashMap<usize, f32>
} 

impl Encoder<'_> {
    fn feedforward(&mut self, word: &str) {
        for i in 0..self.vec_size {
            &self.hidden.insert(i, 1f32);
        }
    }
}
fn main() {

    let mut weights: HashMap<(&str, usize), Nexus> = HashMap::new();
    let word = String::from("cat");
    for i in 0..300 {
        weights.insert( (&word, i), Nexus::new());
    }

    
    println!("Hello, world!");
}
