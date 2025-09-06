from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

class LightTransformerModel(nn.Module):
    def __init__(self, model_name, save_path=None):
        """
        Initializes the LightTransformerModel with a specified transformer model.
        Suggested model names are 'distilbert-base-uncased', 'albert-base-v2', 'roberta-base'.
        """

        super(LightTransformerModel, self).__init__()
        self.model_name = model_name
        self.save_path = save_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        #add adapter layers

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    #optimizer = AdamW(model.parameters(), lr=5e-5)
    def train_model(self, dataloader, optimizer, loss_function= nn.CrossEntropyLoss, epochs=10, device=None, save_model=True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        for epoch in tqdm(range(epochs)):
            self.model.train()
            # for batch in dataloader:
            #     input_ids, attention_mask, labels = batch
            #     input_ids = input_ids.to(device)
            #     attention_mask = attention_mask.to(device)
            #     labels = labels.to(device)
            #
            #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            #     loss = loss_function(outputs, labels)
            #
            #     loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()
            
            print(f"Epoch {epoch+1}/{epochs} completed.")
        
        if save_model:
            self._save_model()

    def _save_model(self):
        if self.save_path is None:
            self.save_path = f"./{self.model_name}"

        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

    def eval_model(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.model.eval()
        #evaluate quantitive metrics (ss, lm)
        #evaluate qualitative metrics (to be defined)
    