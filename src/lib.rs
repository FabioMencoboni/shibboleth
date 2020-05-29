
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

fn blazing_dot(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    // a fast, parallel dot product
    // see https://github.com/ChingChuan-Chen/dot_product/blob/master/rust_dotproduct/src/main.rs
    let res: f64 = x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).sum();
    res
}

fn blazing_dist(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    // a fast, parallel vector distance
    let res: f64 = x.par_iter().zip(y.par_iter()).map(|(a, b)| (a-b)*(a-b)).sum();
    res.sqrt()
}

fn sigmoid(z: &f64) -> f64 {
    let a:f64 = 1f64/(1f64+2.7182818f64.powf(-z));
    a
}


fn vec_update(a: &mut Vec<f64>, b: &mut Vec<f64>, alpha: f64) {
    // update vec a by adding alpha * vec b
    for (ai, bi) in a.iter_mut().zip(b) {
        *ai += (alpha * *bi);
    }
}

fn mutvec_mxabc(m: f64, x: &mut Vec<f64>, a: f64, b: &Vec<f64>, c: f64) {
    // mutate the vector x with another vector b and three constants m,a,c:
    // <x> = m*<x> +a*<b> +c
    for (xi, bi) in x.iter_mut().zip(b) {
        *xi = (m * *xi) + (a * *bi) + c
    }
}



pub struct Encoder {
    vec_size: usize,                        // dimensionality of word embeddings
    pub alpha: f64,
    ct_epochs: f64,                          // # of training ephocs completed
    ct_docs: f64,
    ct_words: f64,
    total_error: f64,
    vocab: HashMap<String, (usize, usize)>,        // word -> (index, frequency)
    word_input_vecs: HashMap<String, Vec<f64>>,  // weights from input word to hidden layer
    word_output_vecs: HashMap<String, Vec<f64>>, // weights from hidden layer to output word
    negative_samples: HashMap<usize, String> ,
    negative_idx: usize
} 

impl Encoder {
    pub fn new(vec_size: usize, vocab_file: &str, alpha: f64) -> Encoder {
        // generate and return a new Encoder using a specified vector size and a vocabulary file
        println!("Initializing a new encoder with {}-element word vectors using {}", &vec_size, &vocab_file);
        let vocab = load_vocab(vocab_file);
        let mut rng = rand::thread_rng();
        let mut win: f64;// = -0.01f64 + 0.02f64*rng.gen::<f64>();
        let mut wout: f64;// = -0.01f64 + 0.02f64*rng.gen::<f64>();
        let mut word_input_vecs: HashMap<String, Vec<f64>> = HashMap::new();
        let mut word_output_vecs: HashMap<String, Vec<f64>> = HashMap::new();
        for (key, (index, count)) in vocab.iter() {
            let mut vec_in: Vec<f64> = Vec::new();
            let mut vec_out: Vec<f64> = Vec::new();
            for j in 0..vec_size {
                let win = -0.5f64 + 0.9f64*rng.gen::<f64>();
                let wout = -0.5f64 + 0.9f64*rng.gen::<f64>();
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
        let enc = Encoder{vec_size: vec_size,alpha:alpha, ct_epochs: 0f64,ct_docs: 0f64,total_error:0f64, ct_words:0f64,vocab: vocab.clone(), word_input_vecs: word_input_vecs, word_output_vecs: word_output_vecs, negative_samples:negative_samples, negative_idx:0};
        enc // return the encoder object
    }

    pub fn word_vec(&self, word: &str) -> Option<&Vec<f64>> {
        // given a word, tokenize it and return its vec if the token has one
        let null_vec: Option<&Vec<f64>> = None;
        let tok = tokenize(word)[0].clone();
        let ovec: &Vec<f64> = match self.word_input_vecs.get(&tok){
            Some(val) => val,
            None => return null_vec
        };
        Some(ovec) // return the vector
    }

    pub fn closest_neighbors(&self, word: &str) -> Option<Vec<(String, f64)>> {
        // take word and tokeize it
        // return the closest 7 neighbors
        let null_neighbors: Option<Vec<(String, f64)>> = None;
        let o_vec: Option<&Vec<f64>> = self.word_vec(word);
        let comp_vec: &Vec<f64> = match o_vec {
            Some(vec) => vec,
            None => {println!("{} does not tokenize to anything in the vocabulary.", &word);
                    return null_neighbors}
        };
        // create a list of distances
        let mut dist: f64;
        let mut dist_list: Vec<(&String, f64)> = Vec::new();
        for (word, vec) in self.word_input_vecs.iter() {
            dist = blazing_dist(comp_vec, vec);
            dist_list.push((word, dist));          
        }
        println!("COMP VEC VEC {:#?} FOR {}", &comp_vec, &word);
        // sort the distances. see https://stackoverflow.com/questions/40091161/sorting-a-vector-of-tuples-needs-a-reference-for-the-second-value
        dist_list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for i in 0..12 {
            println!("{} -> '''{}''' dist = {}", word, dist_list[i].0, dist_list[i].1);
        }
        null_neighbors

    }

    pub fn predict(&self, x_input_word: &str, y_output_word: &str) -> Option<f64> {
        // give the sigmoid (not softmax) output for an input and and output string
        // the Option<f64> will be of the None type if one of the keys is missing
        let null_32: Option<f64> = None; 
        let input_tok = tokenize(x_input_word)[0].clone();
        let output_tok = tokenize(y_output_word)[0].clone();
        let x_vec: &Vec<f64> = match self.word_input_vecs.get(&input_tok){
            Some(val) => val,
            None => return null_32
        };
        let y_vec: &Vec<f64> = match self.word_output_vecs.get(&output_tok){
            Some(val) => val,
            None => return null_32
        };
        let z: f64 = blazing_dot( x_vec,  y_vec);
        // apply sigmoid activation
        let a: f64 = sigmoid(&z);
        Some(a)
    }


    fn example(&mut self, input_tok: &str, outputs: HashMap<String, f64>) -> Option<f64> {
        // train on one window + some negative samples example
        // input_word: input string
        // output: String-> 1f64 for grams in the window, String->0f64 
        let null_32: Option<f64> = None; 
        let mut ct_outputs: f64 = 0f64;
        let mut squared_error: f64 = 0f64;
        let mut hidden_error: Vec<f64> = vec![0f64; self.vec_size]; // "error" at the hidden layer
        if ! &self.word_input_vecs.contains_key(input_tok){
            return null_32
        }
        let x_vec = self.word_input_vecs.get_mut(input_tok).unwrap();
        for (output_tok, is_in_window) in outputs.iter() {
            
            if ! self.word_output_vecs.contains_key(output_tok){
                continue
            }
            //let word_freq = self.vocab.get(output_tok).unwrap();
            //println!("EXAMPLE:{} -> {} val={} freq={:#?}", &input_tok, &output_tok, is_in_window, word_freq);
            ct_outputs = ct_outputs + 1f64;
            let y_vec = self.word_output_vecs.get_mut(output_tok).unwrap();
            let z: f64 = blazing_dot(x_vec, y_vec);
            let a: f64 = sigmoid(&z);// apply sigmoid activation
            let word_error: f64 = a - is_in_window;
            //println!("wdERR. {}", &word_error);
            //vec_delta( y_vec, -0.05f64*word_error);
            squared_error = squared_error + (word_error * word_error);
            // START OF GRADIENT DESCENT MATHS
            
            //vec_update(&mut hidden_error,  y_vec, word_error);
            mutvec_mxabc(1f64, &mut hidden_error, word_error, y_vec, 0f64);
            //vec_update(y_vec, x_vec, -self.alpha*word_error);
            mutvec_mxabc(0.99f64, y_vec, -self.alpha*word_error, x_vec, 0f64);
            
        }
        //vec_update(x_vec, &mut hidden_error, -self.alpha/18f64); // about 18 docs
        mutvec_mxabc(0.99f64, x_vec, -self.alpha/ct_outputs, &hidden_error, 0f64);
        //println!("HedERRRRR {:#?}", &hidden_error);
        // END OF GRADIENT DESCENT MATHS
        Some(squared_error/ct_outputs)
    }

    pub fn train_doc(&mut self, document: &str) {
        let mut start: usize = 0;
        let mut end: usize = 0;
        let tokens = tokenize(document);
        let mut output: HashMap<String, f64>;
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
                output.insert(negative_word, 0f64);
            }       
            if center > 4 { // trying to compare 1-4 on usize (positive only) gives an overflow
                start = center - 4;
            } else {
                start = 0;
            }
            end = cmp::min(tokens.len(), center + 4);
            for position in start..end {
                if position != center {
                    output.insert(tokens[position].clone(), 1f64);
                }
            }           
            let err = self.example(&tokens[center], output);
            self.total_error = self.total_error + match err {
                Some(val) => val,
                None => 0f64
            };
            self.ct_words = self.ct_words+1f64;
            
        }
        self.ct_docs = self.ct_docs+1f64;
        println!("ct_docs={}, ct_words={}, tot_err/ct_docs={}", self.ct_docs, self.ct_words, self.total_error/self.ct_docs);
    }


    pub fn train_from_db(&mut self, db_file: &str, n_docs: usize, skip_docs: usize) {
    
        let mut rng = rand::thread_rng();
        let mut docs_processed: usize = 0;
        let mut doc_errors = 0f64;

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
    let mut enc = Encoder::new(40, "WikiVocab25k.txt", 0.06f64);
    let mut activation: f64 = 0f64;
    for _ in 0..100 {
        enc.train_doc("I like to eat fish & chips.");
        enc.train_doc("Steve has chips with his fish.");
        let p: Option<f64> = enc.predict("fish", "chips");
        activation = match p {
            Some(val) => val,
            None => 0f64
        };
        println!("activation {}", activation);
    }
    let p: Option<f64> = enc.predict("fish", "chips");
    activation = match p {
        Some(val) => val,
        None => 0f64
    };
    println!("activation {}", activation);
    assert!(activation > 0.93);
}

//#[test]
fn mini_trial_training(){
    let mut enc = Encoder::new(40, "WikiVocab25k.txt", 0.05);
    for _ in 0..20{
        enc.train_doc("Steve has chips with his fish.");
    }
}

//#[test]
fn trial_training(){
    let mut enc = Encoder::new(40, "WikiVocab25k.txt", 0.05);
    enc.train_from_db("wiki.db", 3, 0);
}