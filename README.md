# shibboleth

### A simple, pure-Rust implementation of word2vec with stemming and negative sampling. With *Shibboleth* you can easily
- **Build** a corpus vocabulary.
- **Train** word vectors
- **Find** words based on vector distance.

#### Automatic text tokenization

```
let tokens = shibboleth::tokenize("Totally! I love cupcakes!");
assert_eq!(tokens[0], "total");
assert_eq!(tokens[3], "cupcak");
```
#### Getting Data In

Shibboleth can use training corpora provided in an sqlite file matching this schema:
```
CREATE TABLE documents (id PRIMARY KEY, text);
```
A popular resource for training purposes is Wikipedia. The script below will download and unzip such a sqlite file with just over 5 million documents. For the wiki license see [here](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content).
```
$ wget -O wiki.db.gz https://dl.fbaipublicfiles.com/drqa/docs.db.gz && gunzip wiki.db.gz
```
#### Building Vocabulary

This example takes the *wiki.db* file downloaded above, runs through the first 1,000,000 documents, stems them, and builds a vocabulary of the 25,000 most common words. The output will be saved to WikiVocab25k.txt
```
use shibboleth;
shibboleth::build_vocab_from_db("wiki.db", "WikiVocab25k.txt", 1000000, 25000);
```

#### Training

```
use shibboleth;

// create a new encoder object with 200 elements per word vector from a vocabulary file
let mut enc = shibboleth::Encoder::new(200, "WikiVocab25k.txt");

// the prediction (sigmoid) for 'chips' occuring near 'fish' should be near 0.5 prior to training
let p = enc.predict("fish", "chips");
match p {
    Some(val) => println!("'Fish'->'Chips' sigmoid activation before training: {}", val),
    None => println!("One of these words is not in your vocabulary")
}

// train on two examples
enc.train_doc("I like to eat fish & chips.");
enc.train_doc("Steve has chips with his fish.");

// after training, the prediction should be near unity
let p = enc.predict("fish", "chips");
match p {
    Some(val) => println!("'Fish'->'Chips' sigmoid activation after training: {}", val),
    None => println!("One of these words is not in your vocabulary")
}
```
Typical Output:
```
'Fish'->'Chips' sigmoid activation before training: 0.5002038
'Fish'->'Chips' sigmoid activation after training: 0.999495
```
