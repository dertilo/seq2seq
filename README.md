# Seq2Seq Dialogue
## datasets

* [Natural Questions](https://ai.google.com/research/NaturalQuestions/dataset) on [github](https://github.com/google-research-datasets/natural-questions)
  + `wget https://storage.googleapis.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz`
* [topical-chat](https://github.com/alexa/alexa-prize-topical-chat-dataset)
* [hotpotqa](https://hotpotqa.github.io/); [hotpot-paper](https://nlp.stanford.edu/pubs/yang2018hotpotqa.pdf)
* [coqa](https://stanfordnlp.github.io/coqa/)
* [SQUAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
* [convai-challenge](http://convai.io/) and [persona-chat](https://github.com/DeepPavlov/convai)
    + https://gist.github.com/thomwolf/ecc52ea728d29c9724320b38619bd6a6
* [quangaroo](http://qangaroo.cs.ucl.ac.uk/)
    + multi-document QA, how to use? concat documents?? Context way to big to fit in transformer
    
#### number of seq2seq-examples    
```
topicalchat-train: 182347
personachat-train: 262875
coqa-train: 108646
squad20-train: 86820
coqa-val: 7982
squad20-val: 20301
```
0. build corpus: `python build_seq2seq_corpus.py`
0. train: `bash run_train.sh 1 2`
1. evaluate: `export PYTHONPATH=$HOME/transformers:$HOME/transformers/examples` and `python evaluate.py`

* scores look like total shit
    ```python
    ### ckpt 0 
    {'rouge-1': {'f': 0.07989701977251337,
                 'p': 0.04469210491383478,
                 'r': 0.5758200577200578},
     'rouge-2': {'f': 0.04629125690555241,
                 'p': 0.02595835462144872,
                 'r': 0.33491416857995804},
     'rouge-l': {'f': 0.10483736259282633,
                 'p': 0.0606928395429648,
                 'r': 0.5718876995553469}}
    
    ### ckpt 1 
    {'rouge-1': {'f': 0.08243292827413572,
                 'p': 0.046208073848713546,
                 'r': 0.5752917388167387},
     'rouge-2': {'f': 0.04764197955935347,
                 'p': 0.02675712171558292,
                 'r': 0.334973912344965},
     'rouge-l': {'f': 0.10797371954306047,
                 'p': 0.06287075612168479,
                 'r': 0.5711747078085313}}
    
    ```
-> maybe scaling down, this time training on coqa only