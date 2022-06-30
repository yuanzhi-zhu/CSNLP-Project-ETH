import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertPreTrainedModel
from tqdm import tqdm, trange

from data.processors.coqa import Extract_Features, Processor, Result
from data.processors.metrics import get_predictions

#   our model is adapted from the baseline model of https://arxiv.org/pdf/1909.10772.pdf

class BertBaseUncasedModel(BertPreTrainedModel):

    #   Initialize Layers for our model
    def __init__(self,config,activation='relu'):
        super(BertBaseUncasedModel, self).__init__(config)
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        self.fc=nn.Linear(hidden_size,hidden_size)
        self.linear1 =nn.Linear(hidden_size,1)
        self.linear2= nn.Linear(hidden_size,2)
        self.activation = getattr(F, activation)
        self.init_weights()

    def forward(self,input_ids,token_type_ids=None,attention_mask=None,start_positions=None,end_positions=None,rational_mask=None,cls_idx=None,head_mask=None):
        #   Bert-base outputs
        outputs = self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,head_mask=head_mask)
        output_vector, bert_pooled_output = outputs

        #   rational logits (rationale probability to calculate start and end logits)
        #   fc = w2 x relu(W1 x h)
        rational_logits = self.fc(output_vector)
        rational_logits = self.activation(self.linear1(rational_logits))

        #   pr = sigmoid(fc)
        rational_logits = torch.sigmoid(rational_logits)
        #   h1 = pr x outputvector-h
        output_vector = output_vector * rational_logits
        mask = token_type_ids.type(output_vector.dtype)
        rational_logits = rational_logits.squeeze(-1) * mask

        #   calculating start and end logits using FC(h1)
        start_end_logits = self.fc(output_vector)
        start_end_logits = self.activation(self.linear2(start_end_logits))
        
        start_logits, end_logits = start_end_logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        start_logits= start_logits * rational_logits
        end_logits =  end_logits * rational_logits

        #   fc2 = wa2 x relu(Wa1 x h1)
        attention  = self.fc(output_vector)
        attention  = (self.activation(self.linear1(attention))).squeeze(-1)

        #   a = SoftMax(fc2)
        attention = F.softmax(attention, dim=-1)
        attention_pooled_output = (attention.unsqueeze(-1) * output_vector).sum(dim=-2)
        unk_logits = self.fc(bert_pooled_output)
        unk_logits = self.activation(self.linear1(unk_logits))

        #   calculate yes and no logits using pooled-output = FC(a)
        yes_no_logits =self.fc(attention_pooled_output)
        yes_no_logits =self.activation(self.linear2(yes_no_logits))
        yes_logits, no_logits = yes_no_logits.split(1, dim=-1)

        if start_positions != None and end_positions != None:
            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx
            start = torch.cat((yes_logits, no_logits, unk_logits, start_logits), dim=-1)
            end = torch.cat((yes_logits, no_logits, unk_logits, end_logits), dim=-1)
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            #   calculate cross entropy loss for start and end logits
            Entropy_loss = CrossEntropyLoss()
            start_loss = Entropy_loss(start, start_positions)
            end_loss = Entropy_loss(end, end_positions)
            #   Training objective: to minimize the total loss of both start and end logits
            total_loss = (start_loss + end_loss) / 2 
            return total_loss
        return start_logits, end_logits, yes_logits, no_logits, unk_logits
        
        
### load dataset for training or evaluation

def load_dataset(tokenizer, input_dir=None, evaluate=True, cache_file_name=None, train_file_name=None, predict_file_name=None, append_method='original'):
    '''
    converting raw coqa dataset into features to be processed by BERT  
    '''
    
    train_file="coqa-train-v1.0.json"
    predict_file="coqa-dev-v1.0.json"

    # print(os.path.join(input_dir,"bert-base-uncased_train"))
    # use user-defined cache_file_name or the default ones
    if cache_file_name is not None:
        cache_file = os.path.join(input_dir,cache_file_name)
    else:
        if evaluate:
            cache_file = os.path.join(input_dir,"bert-base-uncased_dev")
        else:
            cache_file = os.path.join(input_dir,"bert-base-uncased_train")

    if os.path.exists(cache_file):
        print("Loading cache",cache_file)
        features_and_dataset = torch.load(cache_file)
        features, dataset, examples = (
            features_and_dataset["features"],features_and_dataset["dataset"],features_and_dataset["examples"])
    else:
        print("Creating features from dataset file at", input_dir)
        if train_file_name is not None:
            train_file = train_file_name
        if predict_file_name is not None:
            predict_file = predict_file_name

        if not "data" and ((evaluate and not predict_file) or (not evaluate and not train_file)):
            raise ValueError("predict_file or train_file not found")
        else:
            processor = Processor()
            if evaluate:
                # process the raw data, load only two historical conversation
                # def get_examples(self, data_dir, history_len, filename=None, threads=1)
                examples = processor.get_examples(input_dir, 2, filename=predict_file, threads=1, append_method=append_method)
            else:
                # process the raw data
                # def get_examples(self, data_dir, history_len, filename=None, threads=1)
                # number of examples is the same as the number of the QA pairs: 108647
                # each example is consist of question_text with 2 historical turn and the text, and ground truth start and end positions
                examples = processor.get_examples(input_dir, 2, filename=train_file, threads=1, append_method=append_method)
        
        # max_seq_length is the total length for input sequence of BERT 
        features, dataset = Extract_Features(examples=examples,tokenizer=tokenizer,max_seq_length=512, doc_stride=128, max_query_length=64, is_training=not evaluate, threads=1)
    #   caching it in a cache file to reduce time
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cache_file)
    if evaluate:
        return dataset, examples, features
    return dataset
    
    
    
### wrtiting predictions with fine-tuned model

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()


def Write_predictions(model, tokenizer, device, variant_name, input_dir=None,output_directory=None,cache_file_name=None,predict_file_name=None,evaluation_batch_size=1,method='', append_method='original'):
    # generate catch file processed from the json dataset
    dataset, examples, features = load_dataset(tokenizer, input_dir=input_dir, evaluate=True, cache_file_name=cache_file_name, predict_file_name=predict_file_name, append_method=append_method)
    
    if not os.path.exists(output_directory+'/'+variant_name):
        os.makedirs(output_directory+'/'+variant_name)
        
    #   wrtiting predictions once training is complete
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=evaluation_batch_size)
    mod_results = []
    for batch in tqdm(evaluation_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            # each batch has 4 elements, the last is the examle_indeces
            inputs = {"input_ids": batch[0],"token_type_ids": batch[1],"attention_mask": batch[2]}
            # indices of ConvQA example in this batch
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [convert_to_list(output[i]) for output in outputs]
            start_logits, end_logits, yes_logits, no_logits, unk_logits = output
            result = Result(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits, yes_logits=yes_logits, no_logits=no_logits, unk_logits=unk_logits)
            mod_results.append(result)

    # Get predictions for development dataset and store it in predictions.json
    output_prediction_file = os.path.join(output_directory+'/'+variant_name, "predictions{}.json".format(method))
    print('save prediction file at: {}'.format(output_prediction_file))
    get_predictions(examples, features, mod_results, 20, 30, True, output_prediction_file, False, tokenizer)