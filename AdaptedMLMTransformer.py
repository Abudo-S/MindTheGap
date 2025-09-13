from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

class AdaptedMLMTransformer(nn.Module):
    def __init__(self, model_name="roberta-base", save_path=None):
        """
        Initializes the LightTransformerModel with a specified transformer model.
        Suggested model names are 'distilbert-base-uncased', 'albert-base-v2', 'roberta-base'.
        """

        super(AdaptedMLMTransformer, self).__init__()
        self.model_name = model_name
        self.save_path = save_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        #init adapter layers
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        token_type_ids are used for NSP to separate between tokens within two sentences.
        [token_type_ids in not used in RoBERTa, Albert and DistilBERT models]
        """
        #add adapter layers

        if self.model_name in ['roberta-base', 'albert-base-v2', 'distilbert-base-uncased']:
            assert(token_type_ids is None)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs
    
    #optimizer = AdamW(model.parameters(), lr=5e-5)
    def train_model(self, data_loader, val_data_loader, optimizer, loss_function= nn.CrossEntropyLoss, epochs=10, device=None, save_model=True):
        """
        Trains and evaluate the model using the provided data loaders, optimizer, and loss function.
        Data loaders should yield batches of (input_ids, attention_mask, labels).
        Important: call this method only if the model has adapter layer.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        for epoch in tqdm(range(epochs)):
            self.model.train()
            # for batch in data_loader:
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
            #self.eval_model(val_data_loader, device)

            print(f"Epoch {epoch+1}/{epochs} completed.")
        
        if save_model:
            self._save_model()

    def _save_model(self):
        if self.save_path is None:
            self.save_path = f"./{self.model_name}"

        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

    def eval_model(self, val_data_loader, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in val_data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
        #evaluate quantitive metrics (ss, lm)
        #evaluate qualitative metrics (to be defined)

        return accuracy_score(true_labels, predictions)
    
    def get_model_metrics_evaluation(self):
        return {'loss': self._train_losses,
                'val_loss': self._test_losses,
                'accuracy': self._train_accuracies,
                'val_accuracy': self._test_accuracies}
    