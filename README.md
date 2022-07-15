# CSNLP-Project-ETH
This repository provides the supplementary code for the paper "Combined Transformer for Conversational Question
Answering" submitted as a course project for Computational Semantics for Natural Language Processing at ETH Zürich.

### Framework

![](images/CSNLP_framework.png)

### Hands-on with BERT pre-trained model with QA
As a reference, we used https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626
You can also directly play with the jupyter notebook *hand_on_bert_QA.ipynb*.

### BERT for ConvQA
Even though the dataset CoQA is in an abstractive rather than extractive manner unlike SQuAD, most of the existing BERT-based models treat it as an extractive question answering task (the BERT predict a pair of start and end positions of the passage to extract answers). In this project, we also take the same manner and focus on question rewriting techniques.

We adapted BERT codes from https://github.com/suryamp97/COQA-using-BERT-Natural-Language-Processing-NLP for ConvQA.

### T5 for question rewriting
This part of codes is adapted from https://github.com/Orange-OpenSource/COQAR.
We want to use the rewritten questions from a fine-tuned T5 as additional input to Bert for question answering.
