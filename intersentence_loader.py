from os import path 
import sys
sys.path.append("..")
import dataloader
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import LabelEncoder

#thresholds used to label intersentences
#maximize this threshold for anti-stereotypical evalution, if we want an anti-stereotypical model,
#but it might be unrealistic for the real world
INTERSENTENCE_ANTISTEREOTYPE_SCORE = 0.5 #0.999 
INTERSENTENCE_STEREOTYPE_SCORE = 0.5 #1e-5
INTERSENTENCE_UNRELATED_SCORE = 1e-5 #0.50

class IntersentenceDataset(Dataset):
    def __init__(self, tokenizer, max_seq_length=128, examples=list(), input_file=None, batch_size=1): 
        self.tokenizer = tokenizer
        filename = input_file
        self.emp_max_seq_length = float("-inf")
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.prepend_text = None

        intersentence_examples = None
        if input_file is not None:
            stereoset = dataloader.StereoSet(filename)
            intersentence_examples = stereoset.get_intersentence_examples()
        else:
            intersentence_examples = examples
        
        self.preprocessed = [] 
        for example in intersentence_examples:
            context = example.context
            if self.prepend_text is not None:
                context = self.prepend_text + context 
            for sentence in example.sentences:
                #encoded_dict = self.tokenizer.encode_plus(text=context, text_pair=sentence.sentence, add_special_tokens=True, max_length=self.max_seq_length, truncation_strategy="longest_first", pad_to_max_length=False, return_tensors=None, return_token_type_ids=True, return_attention_mask=True, return_overflowing_tokens=False, return_special_tokens_mask=False) 
                
                encoded_dict = tokenizer(
                    context,
                    sentence.sentence,
                    add_special_tokens=True,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
                
                sentence_label = -1
                if sentence.gold_label == 'stereotype':
                    sentence_label = INTERSENTENCE_STEREOTYPE_SCORE
                elif sentence.gold_label == 'anti-stereotype':
                    sentence_label = INTERSENTENCE_ANTISTEREOTYPE_SCORE
                else: #unrelated
                    sentence_label = INTERSENTENCE_UNRELATED_SCORE

                input_ids = encoded_dict['input_ids']
                token_type_ids = encoded_dict['token_type_ids'] if self.tokenizer.__class__.__name__ == "BertTokenizer" else []
                attention_mask = encoded_dict['attention_mask']
                self.preprocessed.append((input_ids, token_type_ids, attention_mask, sentence.ID, sentence_label))

        print(f"Maximum sequence length found: {self.emp_max_seq_length}")
         
    def __len__(self):
        return len(self.preprocessed) 

    def __getitem__(self, idx):
        input_ids, token_type_ids, attention_mask, sentence_id, sentence_label = self.preprocessed[idx]
        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, token_type_ids, attention_mask, sentence_id, sentence_label

    def _tokenize(self, context, sentence):
        # context = "Q: " + context
        context_tokens = self.tokenizer.tokenize(context)
        context_tokens = [self.tokenizer.convert_tokens_to_ids(i) for i in context_tokens]

        # sentence = "A: " + sentence
        sentence_tokens = self.tokenizer.tokenize(sentence)
        if self.batch_size>1:
            if (len(sentence_tokens) + len(context_tokens)) > self.emp_max_seq_length:
                self.emp_max_seq_length = (len(sentence_tokens) + len(context_tokens))
            while (len(sentence_tokens) + len(context_tokens)) < self.max_seq_length:
                sentence_tokens.append(self.tokenizer.pad_token)
        sentence_tokens = [self.tokenizer.convert_tokens_to_ids(i) for i in sentence_tokens] 

        input_ids = self.add_special_tokens_sequence_pair(context_tokens, sentence_tokens) 
        if self.batch_size>1:
            input_ids = input_ids[:self.max_seq_length]
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        
        # get the position ids
        position_offset = input_ids.index(sep_token_id) 
        assert position_offset>0
        position_ids = [1 if idx>position_offset else 0 for idx in range(len(input_ids))] 
        return input_ids, position_ids

    def add_special_tokens_sequence_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A RoBERTa sequence pair has the following format: <s> A </s></s> B </s>
        """
        sep = [self.tokenizer.sep_token_id]
        cls = [self.tokenizer.cls_token_id]
        if self.tokenizer.__class__.__name__=="XLNetTokenizer":
            return token_ids_0 + sep + token_ids_1 + sep + cls
        elif self.tokenizer.__class__.__name__=="RobertaTokenizer":
            return cls + token_ids_0 + sep + sep + token_ids_1 + sep
        elif self.tokenizer.__class__.__name__=="BertTokenizer":
            return cls + token_ids_0 + sep + token_ids_1 + sep
