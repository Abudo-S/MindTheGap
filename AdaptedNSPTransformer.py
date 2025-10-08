from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from intersentence_loader import IntersentenceDataset
import os

LEARNING_RATE = 5e-5 #0.00005
LORA_DROPOUT_REGULARIZATION = 0.1 #[default=0.1] incresed in case of overfitting, decreased in case of underfitting

class AdaptedNSPTransformer(nn.Module):
    def __init__(self, model_name="roberta-base", use_adapter=False, num_labels=2):
        """
        Initializes the AdaptedNSPTransformer with a specified transformer model.
        Suggested model names are 'distilbert-base-uncased', 'albert-base-v2', 'roberta-base'.
        num_labels: the number of output labels for the classification task (2 for NSP) (3 for NLI).
        """
        super(AdaptedNSPTransformer, self).__init__()
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model_name = f"adapted_{model_name}" if use_adapter else model_name
        self.use_adapter = use_adapter

        #init adapter layer (LoRA)
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, #for NSP
            r=8,  #impacts the size of matrices (d.r) in the adapter layer
            lora_alpha=16,
            target_modules=["query", "value"], #attention layers to adapt
            lora_dropout=LORA_DROPOUT_REGULARIZATION, #regularization
        )

        #combine the configured adapter with the main model
        #note that adapted_model is just a wrapper that points to base model in the memory
        self.adapted_model = get_peft_model(self.model, self.peft_config)
        
        if self.use_adapter:
            print("Trainable parameters in the NSP adapted model:")
            self.adapted_model.print_trainable_parameters()
            #print([(name, param.requires_grad) for name, param in self.model.classifier.named_parameters()])
           
            # #unfreeze the classification head parameters to be trained
            # for name, param in self.adapted_model.classifier.named_parameters():
            #     #if "original_module" not in name:
            #     param.requires_grad = True
        else:
            #freeze pre-trained transformer parameters since we only want to train only the NSP head
            #they won't be updated during backpropagation
            for param in self.model.parameters():
                param.requires_grad = False
            
            #unfreeze the classification head parameters to be trained
            for name, param in self.model.classifier.named_parameters():
                if "original_module" not in name:
                    param.requires_grad = True
        
    def forward(self, input_ids, attention_mask):
        """
        token_type_ids are used for NSP to separate between tokens within two sentences.
        [token_type_ids are not used in RoBERTa, Albert and DistilBERT models]
        """
        #use adapter layers
        outputs = None
        if self.use_adapter:
            outputs = self.adapted_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs
    
    def train_epoch(self, 
                    train_dataset : IntersentenceDataset,
                    optimizer=None,
                    lr_scheduler=None,
                    loss_fn= nn.MSELoss,
                    batch_size=5,
                    device=None):
        loss_fn = loss_fn()
        if optimizer is None:
            optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # for batch_num, batch in tqdm(train_loader, len(train_loader)):
        #     print(f"Batch size: {len(batch[0])}")

        losses = []
        for input_ids, token_ids, attention_mask, sentence_ids, sentence_labels in train_loader:
            optimizer.zero_grad()

            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)

            outputs = None
            if self.use_adapter:
                self.adapted_model.train()
                self.adapted_model.to(device)
                outputs = self.adapted_model(input_ids=input_ids,attention_mask=attention_mask)
            else:
                self.model.train()
                self.model.to(device)
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

            if lr_scheduler is not None:
                lr_scheduler.step() #lr hypterparameter tuning during training

            losses.append(loss.item())

        return sum(losses) / len(losses)

    def save_nsp_layer(self, target_path="saved_models/nsp_classification_layer.pt"):
        classifier_state_dict = self.model.classifier.state_dict()

        #save the state_dict
        torch.save(classifier_state_dict, target_path)

        print(f"Saved {self.model.__class__.__name__} classification layer in {target_path}")
    
    def load_nsp_layer(self, target_path="saved_models/nsp_classification_layer.pt"):
        assert os.path.exists(target_path), f"Error: The path '{target_path}' does not exist."

        #load the state_dict into the model's classification layer
        state_dict = torch.load(target_path)
        self.model.classifier.load_state_dict(state_dict, strict=True)

        print(f"Loaded {self.model.__class__.__name__} classification layer from {target_path}")
    
    def save_adapter_layer(self, target_path="saved_models/nsp_adapter"):
        assert os.path.exists(target_path), f"Error: The path '{target_path}' does not exist."

        #save the adapter
        self.adapted_model.save_pretrained(target_path)

        print(f"Saved {self.model.__class__.__name__} adapter layer in {target_path}")
    

    def load_adapter_layer(self, target_path="saved_models/nsp_adapter"):
        assert os.path.exists(target_path), f"Error: The path '{target_path}' does not exist."

        #load the adapter with the base model
        self.adapted_model = PeftModel.from_pretrained(self.model, target_path)

        print(f"Loaded {self.model.__class__.__name__} adapter layer from {target_path}")
    

