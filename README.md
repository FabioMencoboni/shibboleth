# shibboleth
A simple, pure-Rust implementation of word2vec embeddings with word stemming and negative sampling.

#### Getting Data In
Shibboleth can use training corpora provided in an sqlite file matching this schema:
```
CREATE TABLE documents (id PRIMARY KEY, text);
```
A popular resource for training purposes is Wikipedia. The script below will download and unzip such a sqlite file with just over 5 million documents. For the wiki license see [here](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content).
```
$ wget -O wiki.db.gz https://dl.fbaipublicfiles.com/drqa/docs.db.gz && gunzip wiki.db.gz
```
