from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from intersentence_loader import IntersentenceDataset

class AdaptedNSPTransformer(nn.Module):
    def __init__(self, model_name="roberta-base", save_path=None, num_labels=2):
        """
        Initializes the AdaptedNSPTransformer with a specified transformer model.
        Suggested model names are 'distilbert-base-uncased', 'albert-base-v2', 'roberta-base'.
        nsp_dim: dimension of the feedforward layer in the classification head, it needs to be tuned.
        It shall be reduced in case of overfitting, and increased in case of underfitting.
        """
        super(AdaptedNSPTransformer, self).__init__()
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        #init adapter layers

        #Freeze pre-trained transformer parameters since we only want to train only the NSP head
        #They won't be updated during backpropagation
        for param in self.model.parameters():
            param.requires_grad = False
        
        #Unfreeze the classification head parameters
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        token_type_ids are used for NSP to separate between tokens within two sentences.
        [token_type_ids in not used in RoBERTa, Albert and DistilBERT models]
        """
        #use adapter layers

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    def train_epoch(self, 
                    train_dataset : IntersentenceDataset,
                    optimizer=None,
                    loss_fn= nn.MSELoss,
                    batch_size=5,
                    device=None):
        loss_fn = loss_fn()
        if optimizer is None:
            optimizer = AdamW(self.model.parameters(), lr=5e-5) #0.00005

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.train()
        self.model.to(device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # for batch_num, batch in tqdm(train_loader, len(train_loader)):
        #     print(f"Batch size: {len(batch[0])}")

        for input_ids, token_ids, attention_mask, sentence_ids, sentence_labels in train_loader:
            optimizer.zero_grad()

            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)

            outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            else:
                outputs = outputs[0]

            outputs = torch.softmax(outputs, dim=1)

            predictions = outputs[:, 1].float().to(device)
            true_labels = sentence_labels.float().to(device)

            # Calculate loss and backpropagate
            loss = loss_fn(predictions, true_labels)
            loss.backward()
            optimizer.step()

