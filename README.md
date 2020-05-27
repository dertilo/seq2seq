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
1. evaluate-rouge: `source activate huggingface && export PYTHONPATH=$HOME/transformers/examples` and `python evaluate.py`
2. evaluate-coqa with [coqa-baselines](https://github.com/stanfordnlp/coqa-baselines)

```shell script
{'bart': {'em': 45.7, 'f1': 63.3, 'turns': 7983},
 'cheatbot': {'em': 94.7, 'f1': 97.3, 'turns': 7983}, # should be at 100 percent! but it is not!
 'echobot': {'em': 0.0, 'f1': 3.5, 'turns': 7983}}
```

# dash-frontend
2. on gunther, get model from hpc: `rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --exclude=.git tilo-himmelsbach@gateway.hpc.tu-berlin.de:/home/users/t/tilo-himmelsbach/data/bart_seq2seq_dialogue_continued/checkpointepoch=2.ckpt ~/data/bart_coqa_seq2seq/`

![dash-frontend](images/dash_frontend.jpeg)