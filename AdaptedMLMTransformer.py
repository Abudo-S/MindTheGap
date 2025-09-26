from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from dataloader import IntrasentenceDataSet
import os

LEARNING_RATE = 5e-5 #0.00005

class AdaptedMLMTransformer(nn.Module):
    def __init__(self, model_name="roberta-base", use_adapter=False):
        """
        Initializes the AdaptedMLMTransformer with a specified transformer model.
        Suggested model names are 'distilbert-base-uncased', 'albert-base-v2', 'roberta-base'.
        """
        super(AdaptedMLMTransformer, self).__init__()
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model_name = f"adapted_{model_name}" if use_adapter else model_name
        self.use_adapter = use_adapter
        
        #init adapter layer (LoRA)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, #for MLM
            r=8, #impacts the size of matrices (d.r) in the adapter layer
            lora_alpha=16,
            target_modules=["query", "value"], #attention layers to adapt
            lora_dropout=0.1, #regularization
        )

        #combine the configured adapter to the main model
        #note that adapted_model is just a wrapper that points the original model in the memory
        self.adapted_model = get_peft_model(self.model, peft_config)

        if self.use_adapter:
            print("Trainable parameters in the MLM adapted model:")
            self.adapted_model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, useAdapter=False):

        #use adapter layers
        outputs = None
        if useAdapter:
            outputs = self.adapted_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs
    
    def train_epoch(self, 
                    train_dataset : IntrasentenceDataSet,
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

        if self.use_adapter:
            self.adapted_model.train()
            self.adapted_model.to(device)
        else:
            self.model.train()
            self.model.to(device)
        
        # for batch_num, batch in tqdm(train_loader, len(train_loader)):
        #     print(f"Batch size: {len(batch[0])}")

        losses = []
        for sentence_ids, _, input_ids, attention_mask, _, sentence_labels in train_loader:
            optimizer.zero_grad()

            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)

            outputs = None
            if self.use_adapter:
                outputs = self.adapted_model(input_ids=input_ids,attention_mask=attention_mask)
            else:
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

    def save_adapter_layer(self, target_path="saved_models/mlm_adapter"):
        assert os.path.exists(target_path), f"Error: The path '{target_path}' does not exist."

        #save the adapter
        self.adapted_model.save_pretrained(target_path)

        print(f"Saved {self.model.__class__.__name__} adapter layer in {target_path}")
    

    def load_adapter_layer(self, target_path="saved_models/mlm_adapter"):
        assert os.path.exists(target_path), f"Error: The path '{target_path}' does not exist."

        #load the adapter with the base model
        self.adapted_model = PeftModel.from_pretrained(self.model, target_path)

        print(f"Loaded {self.model.__class__.__name__} adapter layer from {target_path}")
    
    