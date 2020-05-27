
use std::collections::HashMap;
use rand::Rng;
use zoea::nlp;
use sqlite as db;
use std::fs::File;
use std::io::{Write};//, BufReader, BufRead, Error};



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
    word_idx: HashMap<usize, &'a str>,      // index words so you can use random sampling
    weights: HashMap<(&'a str, usize), Nexus>, // weights
} 

impl Encoder<'_> {
    fn feedforward(&mut self, input: &str, output: &str) {

        for i in 0..self.vec_size {
            println!("");
        }
    }
}


pub fn build_vocab_from_db(db_file: &str, out_file: &str, vocab_size: usize) {
    // build vocabulary from database db_file
    // keep the top vocab_size terms and save to out_file
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let conn = db::open(&db_file).unwrap();
    conn.iterate("SELECT text FROM documents LIMIT 30",	|pairs| {
		for &(_, value) in pairs.iter() { // _ = column
			// build a list then use it below, as you can't borrow twice
			let document: String = value.unwrap().to_string();
            let tokens = nlp::text_tokens(&document);
            for tok in tokens {
                println!("{}", tok);
            }
        } true
    }).unwrap();
}
/**
 let conn = db::open(&db_name).unwrap();
self.conn
			.iterate(expr,
				|pairs| {
				for &(_, value) in pairs.iter() { // _ = column
					// build a list then use it below, as you can't borrow twice
					let firstword: String = value.unwrap().to_string();
					set_firstwords.insert(firstword);

				} true
		   }).unwrap();
 */
fn main() {

    let mut weights: HashMap<(&str, usize), Nexus> = HashMap::new();
    let word = String::from("cat");
    for i in 0..300 {
        weights.insert( (&word, i), Nexus::new());
    }

    let sentence = String::from("Today I walked slowly to the garden in San Diego.");
    let tokens = nlp::text_tokens(&sentence);
    for tok in tokens {
        println!("{}", tok);
    }
    build_vocab_from_db("wiki.db","TBD",20);

    let path = "lines.txt";

    let mut output = File::create(path).unwrap();
    write!(output, "Rust\n💖\nFun").unwrap();
    println!("Hello, world!");
}
