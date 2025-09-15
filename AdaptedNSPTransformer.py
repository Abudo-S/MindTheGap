from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

class AdaptedNSPTransformer(nn.Module):
    def __init__(self, model_name="roberta-base", save_path=None, num_labels=2):
        """
        Initializes the AdaptedNSPTransformer with a specified transformer model.
        Suggested model names are 'distilbert-base-uncased', 'albert-base-v2', 'roberta-base'.
        nsp_dim: dimension of the feedforward layer in the classification head, it needs to be tuned.
        It shall be reduced in case of overfitting, and increased in case of underfitting.
        """
        super(AdaptedNSPTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        #init adapter layers

        # #Freeze pre-trained transformer parameters since we only want to train only the adapter layers
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        token_type_ids are used for NSP to separate between tokens within two sentences.
        [token_type_ids in not used in RoBERTa, Albert and DistilBERT models]
        """
        #use adapter layers

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
