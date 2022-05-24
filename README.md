# CSNLP-Project-ETH
This is the repo for our computational semantics natural language processing course project (2022F)

### Hand on with BERT pre-trained model with QA
https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626
You can also directly play with the jupyter notebook *hand_on_bert_QA.ipynb*.

### BERT for ConvQA

Even though the dataset CoQA is in an abstractive rather than extractive manner unlike SQuAD, most of the existing BERT-based models treat it as an extractive question answering task (the BERT predict a pair of start and end positions of the passage to extract answers). In this project, we also take the same manner and focus on question rewriting techniques.

### BERT codes
We adapted bert codes from https://github.com/suryamp97/COQA-using-BERT-Natural-Language-Processing-NLP.

In the future, we will construct our own input for this BERT.


### T5 for question rewriting
This part of codes is adapted from https://github.com/Orange-OpenSource/COQAR.

We want to use the rewritten question from a fine-tuned T5 as additional input to Bert for question answering.

### Framework

![](images/CSNLP_framework.png)
