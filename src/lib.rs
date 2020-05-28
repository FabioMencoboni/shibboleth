
use std::collections::HashMap;
use rand::{Rng};
use zoea::nlp;
use sqlite as db;
use std::{cmp, fs::File, io::{self, BufReader,Write}};
use std::io::prelude::*; // needed to have File.lines() work


pub fn tokenize(text: &str) -> Vec<String> {
    let tokens = nlp::text_tokens(text);
    tokens
}

pub struct Encoder {
    vec_size: usize,                        // dimensionality of word embeddings
    ct_epochs: f32,                          // # of training ephocs completed
    ct_docs: f32,
    ct_words: f32,
    total_error: f32,
    vocab: HashMap<String, (usize, usize)>,              // word -> (index, frequency)
    w_in_to_hidden: HashMap<(usize, usize), f32>,
    w_hidden_to_out: HashMap<(usize, usize), f32>, // weights // weights
    negative_samples: HashMap<usize, String> ,
    negative_idx: usize
} 

impl Encoder {
    pub fn new(vec_size: usize, vocab_file: &str) -> Encoder {
        // generate and return a new Encoder using a specified vector size and a vocabulary file
        println!("Initializing a new encoder with {}-element word vectors using {}", &vec_size, &vocab_file);
        let vocab = load_vocab(vocab_file);
        let mut rng = rand::thread_rng();
        let mut win: f32;// = -0.01f32 + 0.02f32*rng.gen::<f32>();
        let mut wout: f32;// = -0.01f32 + 0.02f32*rng.gen::<f32>();
        let mut w_in_to_hidden: HashMap<(usize, usize), f32> = HashMap::new();
        let mut w_hidden_to_out: HashMap<(usize, usize), f32> = HashMap::new();
        for (key, (index, count)) in vocab.iter() {
            for j in 0..vec_size {
                win = -0.01f32 + 0.02f32*rng.gen::<f32>();
                wout = -0.01f32 + 0.0f32*rng.gen::<f32>();
                w_in_to_hidden.insert((*index, j), win);
                w_hidden_to_out.insert((*index, j), wout);
            }
        }
        // create a lookup table for negative samples
        let mut negative_samples: HashMap<usize, String> = HashMap::new();
        let mut k = 0;
        for (word, _) in &vocab {
            k = k+1;
            negative_samples.insert(k, word.clone());
        }
        let enc = Encoder{vec_size: vec_size, ct_epochs: 0f32,ct_docs: 0f32,total_error:0f32, ct_words:0f32,vocab: vocab.clone(), w_in_to_hidden: w_in_to_hidden, w_hidden_to_out: w_hidden_to_out, negative_samples:negative_samples, negative_idx:0};
        enc // return the encoder object
    }

    pub fn predict(&self, input_word: &str, output: &str) -> Option<f32> {
        // give the sigmoid (not softmax) output for an input and and output string
        // the Option<f32> will be of the None type if one of the keys is missing
        let null_32: Option<f32> = None; 
        let input_tok = tokenize(input_word)[0].clone();
        let output_tok = tokenize(output)[0].clone();
        let input_idx: usize = match self.vocab.get(&input_tok){
            Some(val) => val.0,
            None => return null_32
        };
        let output_idx: usize = match self.vocab.get(&output_tok){
            Some(val) => val.0,
            None => return null_32
        };
        let mut z: f32 = 0f32;
        let mut win: f32;
        let mut wout: f32;
        for j in 0..self.vec_size {
            win = match self.w_in_to_hidden.get(&(input_idx, j)) {
                Some(&val) => val,
                None => 0f32
            };
            wout = match self.w_hidden_to_out.get(&(output_idx, j)) {
                Some(&val) => val,
                None => 0f32
            };
            z = z + (win * wout);
        }
        // apply sigmoid activation
        let a: f32 = 1f32/(1f32+2.718f32.powf(z));
        Some(a)
    }


    pub fn example(&mut self, input_word: &str, outputs: HashMap<String, f32>) -> Option<f32> {
        // train on one window + some negative samples example
        // input_word: input string
        // output: String-> 1f32 for grams in the window, String->0f32 
        let null_32: Option<f32> = None; 
        let mut squared_error: f32 = 0f32;
        let mut hidden_error: HashMap<usize, f32> = HashMap::new();
        let input_idx: usize = match self.vocab.get(input_word){
            Some(val) => val.0,
            None => return null_32 // you gave an input_word not in the vocab
        };
        for (output_word, is_in_window) in outputs.iter() {
            let output_idx: usize = match self.vocab.get(output_word){
                Some(val) => val.0,
                None => continue
            };
            let mut z: f32 = 0f32;
            let mut win: f32;
            let mut wout: f32;
            for j in 0..self.vec_size {
                win = match self.w_in_to_hidden.get(&(input_idx, j)) {
                    Some(&val) => val,
                    None => 0f32 // should never be used
                };
                wout = match self.w_hidden_to_out.get(&(output_idx, j)) {
                    Some(&val) => val,
                    None => 0f32 // should never be used
                };
                z = z + (win * wout);
            }
            // apply sigmoid activation
            let a: f32 = 1f32/(1f32+2.718f32.powf(-z));
            let word_error: f32 = a - is_in_window;
            squared_error = squared_error + (word_error * word_error);
            // update weights from the hidden layer to the output layer
            for j in 0..self.vec_size {
                let wout = match self.w_hidden_to_out.get(&(output_idx, j)) {
                    Some(val) => val.clone(),
                    None => 0f32
                };
                let new_wout: f32 = wout - 0.05f32*word_error;
                self.w_hidden_to_out.insert((output_idx, j), new_wout);
                *hidden_error.entry(j).or_insert(0f32) += (wout*word_error);
            }
        // update weights from the input layer to the hidden layer
        for j in 0..self.vec_size {
            let node_error: f32 = match hidden_error.get(&j){
                Some(val) => val.clone(),
                None => 0f32
            };
            let win = match self.w_in_to_hidden.get(&(input_idx, j)) {
                Some(val) => val.clone(),
                None => 0f32
            };
            //println!("node error {} {}", input_word, &node_error);
            let new_win: f32 = win - 0.05f32*word_error;
            self.w_in_to_hidden.insert((input_idx, j), new_win);
        }
        }
        Some(squared_error)
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
    let mut enc = Encoder::new(200, "WikiVocab25k.txt");
    enc.train_doc("I like to eat fish & chips.");
    enc.train_doc("Steve has chips with his fish.");
    let p: Option<f32> = enc.predict("fish", "chips");
    let activation: f32 = match p {
        Some(val) => val,
        None => 0f32
    };
    assert!(activation > 0.98);
}