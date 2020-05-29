
// READ THIS
// https://stackoverflow.com/questions/30414424/how-can-i-update-a-value-in-a-mutable-hashmap

use std::collections::HashMap;
use rand::{Rng};
use zoea::nlp;
use sqlite as db;
use std::{cmp, fs::File, io::{self, BufReader,Write}};
use std::io::prelude::*; // needed to have File.lines() work
// the next few lines are needed for blazing_dot
use rayon::prelude::*;
use std::vec::Vec;


pub fn tokenize(text: &str) -> Vec<String> {
    let tokens = nlp::text_tokens(text);
    tokens
}

fn blazing_dot(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    // see https://github.com/ChingChuan-Chen/dot_product/blob/master/rust_dotproduct/src/main.rs
    let res: f32 = x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).sum();
    res
}

fn sigmoid(z: &f32) -> f32 {
    let a:f32 = 1f32/(1f32+2.7182818f32.powf(-z));
    a
}


fn vec_update(a: &mut Vec<f32>, b: &mut Vec<f32>, alpha: f32) {
    // update vec a by adding alpha * vec b
    for (ai, bi) in a.iter_mut().zip(b) {
        *ai += (alpha * *bi);
    }
}



pub struct Encoder {
    vec_size: usize,                        // dimensionality of word embeddings
    alpha: f32,
    ct_epochs: f32,                          // # of training ephocs completed
    ct_docs: f32,
    ct_words: f32,
    total_error: f32,
    vocab: HashMap<String, (usize, usize)>,              // word -> (index, frequency)
    word_input_vecs: HashMap<String, Vec<f32>>,  // weights from input word to hidden layer
    word_output_vecs: HashMap<String, Vec<f32>>, // weights from hidden layer to output word
    negative_samples: HashMap<usize, String> ,
    negative_idx: usize
} 

impl Encoder {
    pub fn new(vec_size: usize, vocab_file: &str, alpha: f32) -> Encoder {
        // generate and return a new Encoder using a specified vector size and a vocabulary file
        println!("Initializing a new encoder with {}-element word vectors using {}", &vec_size, &vocab_file);
        let vocab = load_vocab(vocab_file);
        let mut rng = rand::thread_rng();
        let mut win: f32;// = -0.01f32 + 0.02f32*rng.gen::<f32>();
        let mut wout: f32;// = -0.01f32 + 0.02f32*rng.gen::<f32>();
        let mut word_input_vecs: HashMap<String, Vec<f32>> = HashMap::new();
        let mut word_output_vecs: HashMap<String, Vec<f32>> = HashMap::new();
        for (key, (index, count)) in vocab.iter() {
            let mut vec_in: Vec<f32> = Vec::new();
            let mut vec_out: Vec<f32> = Vec::new();
            for j in 0..vec_size {
                win = -0.001f32 + 0.002f32*rng.gen::<f32>();
                wout = -0.001f32 + 0.00f32*rng.gen::<f32>();
                vec_in.push(win);
                vec_out.push(wout);
            word_input_vecs.insert(key.clone(), vec_in.clone());
            word_output_vecs.insert(key.clone(), vec_out.clone());
            }
        }
        // create a lookup table for negative samples
        let mut negative_samples: HashMap<usize, String> = HashMap::new();
        let mut k = 0;
        for (word, _) in &vocab {
            k = k+1;
            negative_samples.insert(k, word.clone());
        }
        let enc = Encoder{vec_size: vec_size,alpha:alpha, ct_epochs: 0f32,ct_docs: 0f32,total_error:0f32, ct_words:0f32,vocab: vocab.clone(), word_input_vecs: word_input_vecs, word_output_vecs: word_output_vecs, negative_samples:negative_samples, negative_idx:0};
        enc // return the encoder object
    }

    pub fn word_vec(&self, word: &str) -> Option<&Vec<f32>> {
        // given a word, tokenize it and return its vec if the token has one
        let null_vec: Option<&Vec<f32>> = None;
        let tok = tokenize(word)[0].clone();
        let ovec: &Vec<f32> = match self.word_input_vecs.get(&tok){
            Some(val) => val,
            None => return null_vec
        };
        Some(ovec) // return the vector
    }

    pub fn predict(&self, x_input_word: &str, y_output_word: &str) -> Option<f32> {
        // give the sigmoid (not softmax) output for an input and and output string
        // the Option<f32> will be of the None type if one of the keys is missing
        let null_32: Option<f32> = None; 
        let input_tok = tokenize(x_input_word)[0].clone();
        let output_tok = tokenize(y_output_word)[0].clone();
        let x_vec: &Vec<f32> = match self.word_input_vecs.get(&input_tok){
            Some(val) => val,
            None => return null_32
        };
        let y_vec: &Vec<f32> = match self.word_output_vecs.get(&output_tok){
            Some(val) => val,
            None => return null_32
        };
        let z: f32 = blazing_dot( x_vec,  y_vec);
        // apply sigmoid activation
        let a: f32 = sigmoid(&z);
        Some(a)
    }


    fn example(&mut self, input_tok: &str, outputs: HashMap<String, f32>) -> Option<f32> {
        // train on one window + some negative samples example
        // input_word: input string
        // output: String-> 1f32 for grams in the window, String->0f32 
        let null_32: Option<f32> = None; 
        let mut ct_outputs: f32 = 0f32;
        let mut squared_error: f32 = 0f32;
        let mut hidden_error: Vec<f32> = vec![0f32; self.vec_size]; // "error" at the hidden layer
        if ! &self.word_input_vecs.contains_key(input_tok){
            return null_32
        }
        let x_vec = self.word_input_vecs.get_mut(input_tok).unwrap();
        for (output_tok, is_in_window) in outputs.iter() {
            if ! self.word_output_vecs.contains_key(output_tok){
                continue
            }
            ct_outputs = ct_outputs + 1f32;
            let y_vec = self.word_output_vecs.get_mut(output_tok).unwrap();
            let z: f32 = blazing_dot(x_vec, y_vec);
            let a: f32 = sigmoid(&z);// apply sigmoid activation
            let word_error: f32 = a - is_in_window;
            //vec_delta( y_vec, -0.05f32*word_error);
            squared_error = squared_error + (word_error * word_error);
            // START OF GRADIENT DESCENT MATHS
            vec_update(y_vec, x_vec, -self.alpha*word_error);
            vec_update(&mut hidden_error,  y_vec, word_error);
        }
        vec_update(x_vec, &mut hidden_error, -self.alpha/18f32); // about 18 docs
        // END OF GRADIENT DESCENT MATHS
        Some(squared_error/ct_outputs)
    }

    pub fn train_doc(&mut self, document: &str) {
        let mut start: usize = 0;
        let mut end: usize = 0;
        let tokens = tokenize(document);
        let mut output: HashMap<String, f32>;
        for center in 0..tokens.len() {
            
            output = HashMap::new();
            // apply negative sampling 
            for _ in 0..10 {
                self.negative_idx = self.negative_idx + 1;
                let neg_mod: usize = self.negative_idx % self.negative_samples.len();
                let negative_word: String = match self.negative_samples.get(&neg_mod) {
                    Some(neg_word) => neg_word.clone(),
                    None => "nonexistttt".to_string()
                };
                output.insert(negative_word, 0f32);
            }       
            if center > 4 { // trying to compare 1-4 on usize (positive only) gives an overflow
                start = center - 4;
            } else {
                start = 0;
            }
            end = cmp::min(tokens.len(), center + 4);
            for position in start..end {
                if position != center {
                    output.insert(tokens[position].clone(), 1f32);
                }
            }           
            let err = self.example(&tokens[center], output);
            self.total_error = self.total_error + match err {
                Some(val) => val,
                None => 0f32
            };
            self.ct_words = self.ct_words+1f32;
            
        }
        self.ct_docs = self.ct_docs+1f32;
        println!("ct_docs={}, ct_words={}, tot_err/ct_docs={}", self.ct_docs, self.ct_words, self.total_error/self.ct_docs);
    }


    pub fn train_from_db(&mut self, db_file: &str, n_docs: usize, skip_docs: usize) {
    
        let mut rng = rand::thread_rng();
        let mut docs_processed: usize = 0;
        let mut doc_errors = 0f32;

        let conn = db::open(&db_file).unwrap();
        conn.iterate(format!("SELECT text FROM documents LIMIT {} OFFSET {}", n_docs, skip_docs),	|pairs| {
            for &(_, value) in pairs.iter() { // _ = column
                // build a list then use it below, as you can't borrow twice
                let document: &str = value.unwrap();
                self.train_doc(document);
       
            } true
        }).unwrap();
    }
}



pub fn load_vocab(vocab_file: &str) -> HashMap<String, (usize, usize)> {
    let mut word = String::new();
    let mut count: usize = 0;
    let mut position: usize = 0;
    let mut vocab: HashMap<String, (usize, usize)> = HashMap::new();
    let mut index: usize = 0;

    let f = File::open(vocab_file).unwrap();
    let f = BufReader::new(f);

    for line in f.lines() {
        let line = line.unwrap();
        let line = line.split_whitespace();
        for x in line {
            position  = position + 1;
            if position % 2 == 1 {
                word = x.to_string();
            } else {
                count = x.parse::<usize>().unwrap();
            }
        }
        vocab.insert(word.to_string(), (index, count));
        index = index + 1;
    }
    vocab
}

pub fn build_vocab_from_db(db_file: &str, vocab_file: &str, n_docs: usize, vocab_size: usize) {
    // build vocabulary from database db_file
    // keep the top vocab_size terms and save to vocab_file
    let mut vocab: HashMap<String, usize> = HashMap::new();
    let mut n_doc: usize = 0;
    let conn = db::open(&db_file).unwrap();
    conn.iterate(format!("SELECT text FROM documents LIMIT {}", n_docs),	|pairs| {
		for &(_, value) in pairs.iter() { // _ = column
			// build a list then use it below, as you can't borrow twice
			let document: &str = value.unwrap();
            let tokens = tokenize(&document);
            for tok in tokens {
                *vocab.entry(tok).or_insert(0) += 1; 
            } 
            n_doc = n_doc + 1;
            if n_doc % 500 == 0 {
                println!("Building vocab...{} docs processed, vocab items={}", n_doc ,vocab.keys().len());
            }
        } true
    }).unwrap();

    let mut counts: Vec<usize> = Vec::new();
    for (_, &count) in vocab.iter() {
        counts.push(count);
    }
    counts.sort();
    counts.reverse();
    let min_freq = counts[cmp::min(vocab_size, counts.len())];

    let mut output = File::create(&vocab_file).unwrap();
    let mut lines_written: usize = 0;
    for (tok, &count) in vocab.iter() {
        if &count >= &min_freq {
            write!(output, "{} {}\n", tok, count).unwrap();
            lines_written = lines_written + 1;
        }
        if lines_written >= vocab_size {
            break
        }
    }
    

}



#[test]
fn test_tokenization(){
    let tokens = tokenize("Totally! I love cupcakes!");
    assert_eq!(tokens[0], "total");
    assert_eq!(tokens[3], "cupcak");
}

#[test]
fn test_sigmoid(){
    let mut enc = Encoder::new(40, "WikiVocab25k.txt", 0.05);
    let mut activation: f32 = 0f32;
    for _ in 0..100 {
        enc.train_doc("I like to eat fish & chips.");
        enc.train_doc("Steve has chips with his fish.");
    }
    let p: Option<f32> = enc.predict("fish", "chips");
    activation = match p {
        Some(val) => val,
        None => 0f32
    };
    println!("activation {}", activation);
    assert!(activation > 0.98);
}

#[test]
fn mini_trial_training(){
    let mut enc = Encoder::new(40, "WikiVocab25k.txt", 0.05);
    for _ in 0..20{
        enc.train_doc("Steve has chips with his fish.");
    }
}

#[test]
fn trial_training(){
    let mut enc = Encoder::new(40, "WikiVocab25k.txt", 0.05);
    enc.train_from_db("wiki.db", 3, 0);
}