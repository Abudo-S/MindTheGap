from transformers import AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from AdaptedMLMTransformer import AdaptedMLMTransformer
from AdaptedNSPTransformer import AdaptedNSPTransformer
import dataloader
from intersentence_loader import IntersentenceDataset
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count

MAX_SEQ_LENGTH = 128
TRAINING_SET_SIZE_PERCENT = 0.85
NO_CUDA = False

class SentenceEvaluator():
    def __init__(self,
                 input_file="data/stereo_dataset.json",
                 model_name="roberta-base",
                 intersentence_load_path=None,
                 intrasentence_load_path=None,
                 skip_intrasentence=False,
                 skip_intersentence=False,
                 batch_size=5,
                 loss_fn = nn.MSELoss):
        print(f"Loading {input_file}...")

        #self.dataloader = dataloader.StereoSet(os.path.abspath(input_file))
        self.input_file = input_file
        self.model_name = model_name
        self.INTRASENTENCE_LOAD_PATH = intrasentence_load_path
        self.INTERSENTENCE_LOAD_PATH = intersentence_load_path
        self.SKIP_INTERSENTENCE = skip_intersentence
        self.SKIP_INTRASENTENCE = skip_intrasentence
        self.batch_size = batch_size
        self.loss_fn = loss_fn()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "cuda" if not NO_CUDA else "cpu"
        self.MASK_TOKEN = self.tokenizer.mask_token
        #saved after first data split
        self.intra_train_dataset = None
        self.intra_test_dataset = None
        self.inter_train_dataset = None
        self.inter_test_dataset = None

        # Set this to be none if you don't want to batch items together!
        self.max_seq_length = None if self.batch_size == 1 else MAX_SEQ_LENGTH

        self.MASK_TOKEN_IDX = self.tokenizer.encode(
            self.MASK_TOKEN, add_special_tokens=False)
        assert len(self.MASK_TOKEN_IDX) == 1
        self.MASK_TOKEN_IDX = self.MASK_TOKEN_IDX[0]

        config = AutoConfig.from_pretrained(self.model_name)
        print(f"Model max position embeddings: {config.max_position_embeddings}")
        print(f"Using device: {self.device}")
        print(f"Using pretrained class: {self.model_name}")
        self._split_datasets()

    def _split_datasets(self):
        '''
        Reproducible split of dataset into train and test sets for both intersentences and
        intrasentences.
        '''
        
        #intrasentences split
        if self.intra_train_dataset is None or self.intra_test_dataset is None:
            intra_examples = dataloader.StereoSet(self.input_file).get_intrasentence_examples()
            gen = torch.Generator().manual_seed(41)
            training_size = int(TRAINING_SET_SIZE_PERCENT * len(intra_examples))
            test_size = len(intra_examples) - training_size
            train_examples, test_examples = random_split(intra_examples, [training_size, test_size], generator=gen)

            train_dataset = dataloader.IntrasentenceDataSet(self.tokenizer, max_seq_length=self.max_seq_length,
                                                    pad_to_max_length='max_length',
                                                    examples=train_examples)
            test_dataset = dataloader.IntrasentenceDataSet(self.tokenizer, max_seq_length=self.max_seq_length,
                                                    pad_to_max_length='max_length',
                                                    examples=test_examples)

            print(f"First element of the [intrasentences] training set: {train_dataset[0]}")
            print(f"First element of the [intrasentances] test set: {test_dataset[0]}")

            #save splitted datasets for future evaluation
            self.intra_train_dataset = train_dataset
            self.intra_test_dataset = test_dataset
        
        #intersentences split
        if self.inter_train_dataset is None or self.inter_test_dataset is None:
            inter_examples = dataloader.StereoSet(self.input_file).get_intersentence_examples()
            gen = torch.Generator().manual_seed(41)
            training_size = int(TRAINING_SET_SIZE_PERCENT * len(inter_examples))
            test_size = len(inter_examples) - training_size
            train_examples, test_examples = random_split(inter_examples, [training_size, test_size], generator=gen)

            train_dataset = IntersentenceDataset(self.tokenizer, max_seq_length=self.max_seq_length,
                                                    examples=train_examples)
            test_dataset = IntersentenceDataset(self.tokenizer, max_seq_length=self.max_seq_length,
                                                    examples=test_examples)

            print(f"First element of the [intersentences] training set: {train_dataset[0]}")
            print(f"First element of the [intersentances] test set: {test_dataset[0]}")
            #save splitted datasets for future evaluation
            self.inter_train_dataset = train_dataset
            self.inter_test_dataset = test_dataset

    def evaluate_intrasentence(self, targetModel=None, useTrainingSet=False):
        model = targetModel.to(self.device) if targetModel else AdaptedMLMTransformer(model_name=self.model_name).to(self.device)

        if torch.cuda.is_available() and self.device == "cuda":
            print("Moving model to GPU...")
            model.to(self.device)
            # Explicitly move the underlying model to the device as well
            if hasattr(model, 'model') and isinstance(model.model, nn.Module):
                model.model.to(self.device)
                #self.tokenizer.to(self.device)
                print(f"Underlying model moved to {self.device}.")
            print(f"{model.__class__.__name__} instance moved to GPU.")
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        else:
            print("CUDA is not available or device is not set to cuda, using CPU.")

        if torch.cuda.device_count() > 1 and self.device == "cuda":
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.eval()

        if self.INTRASENTENCE_LOAD_PATH:
            state_dict = torch.load(self.INTRASENTENCE_LOAD_PATH)
            model.load_state_dict(state_dict)

        #pad_to_max_length = True if self.batch_size > 1 else False

        #dataset = dataloader.IntrasentenceDataSet(self.tokenizer, self.input_file)
        #print(f'Sentence 0: {dataset.sentences[0]}')

        data_loader = DataLoader(self.intra_train_dataset if useTrainingSet else self.intra_test_dataset, batch_size=self.batch_size)
        word_probabilities = defaultdict(list)

        print("Calculating intrasentence predictions...")

        with torch.no_grad():
            for sentence_id, next_token, input_ids, attention_mask, token_type_ids, sentence_label in tqdm(data_loader, total=len(data_loader)):
                if self.batch_size == 1:
                    print(f"Max attention mask value: {max(attention_mask)}")
                    print(f"Min attention mask value: {min(attention_mask)}")
                    max_id = max(input_ids)
                    print(f"Max input ID: {max_id}, Model vocab size: {self.tokenizer.vocab_size}")

                input_ids = input_ids.squeeze(1).to(self.device)
                attention_mask = attention_mask.squeeze(1).to(self.device)
                next_token = next_token.to(self.device) #token to predict

                mask_idxs = (input_ids == self.MASK_TOKEN_IDX)

                # get the probabilities
                output = model(input_ids, attention_mask=attention_mask)[0].softmax(dim=-1)

                output = output[mask_idxs] #target only the masked positions
                output = output.index_select(1, next_token).diag() #extract the probs of true tokens from the vocabulary dimension
                for idx, item in enumerate(output):
                    word_probabilities[sentence_id[idx]].append((item.item(), sentence_label[idx]))

        # now reconcile the probabilities into sentences
        sentence_probabilties = []
        for k, v in word_probabilities.items():
            pred = {}
            pred['id'] = k
            #since we have n next tokens for the same sentance id, associated probs needs to be standarized
            #in order to be compared with other labeled sentences' scores
            v_scores = [v_k for v_k, _ in v]
            # score = np.sum([np.log2(i) for i in v]) + np.log2(len(v))
            score = np.mean(v_scores)
            pred['score'] = score

            loss = self.loss_fn(torch.tensor(v_scores), torch.tensor([v_v for _, v_v in v]))
            pred['loss'] = loss.item()

            sentence_probabilties.append(pred)

        return sentence_probabilties

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_intersentence(self, targetModel=None, useTrainingSet=False):
        model = targetModel.to(self.device) if targetModel else AdaptedNSPTransformer(model_name=self.model_name).to(self.device)

        if torch.cuda.is_available() and self.device == "cuda":
            print("Moving model to GPU...")
            model.to(self.device)
            # Explicitly move the underlying model to the device as well
            if hasattr(model, 'model') and isinstance(model.model, nn.Module):
                model.model.to(self.device)
                #self.tokenizer.to(self.device)
                print(f"Underlying model moved to {self.device}.")
            print(f"{model.__class__.__name__} instance moved to GPU.")
        else:
            print("CUDA is not available or device is not set to cuda, using CPU.")


        #print(f"Number of parameters: {self.count_parameters(model):,}")
        model = torch.nn.DataParallel(model)

        if self.INTERSENTENCE_LOAD_PATH:
            model.load_state_dict(torch.load(self.INTERSENTENCE_LOAD_PATH))

        model.eval()
        #dataset = IntersentenceDataset(self.tokenizer)
        #print(f'Sentence 0: {dataset.sentences[0]}')

        data_loader = DataLoader(self.inter_train_dataset if useTrainingSet else self.inter_test_dataset, batch_size=self.batch_size)

        print("Calculating intersentence predictions...")
        if NO_CUDA:
            n_cpus = cpu_count()
            print(f"Using {n_cpus} cpus!")
            predictions = Parallel(n_jobs=n_cpus, backend="multiprocessing")(delayed(process_job)(
                batch, model, self.model_name) for batch in tqdm(data_loader, total=len(data_loader)))
        else:
            predictions = []

            with torch.no_grad():
                for batch_num, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                    input_ids, token_type_ids, attention_mask, sentence_id, sentence_label = batch
                    input_ids = input_ids.squeeze(1).to(self.device)
                    attention_mask = attention_mask.squeeze(1).to(self.device)
                    #token_type_ids = token_type_ids.to(self.device)

                    outputs = model(input_ids=input_ids,attention_mask=attention_mask)
                    if hasattr(outputs, "logits"):
                        outputs = outputs.logits
                    else:
                        outputs = outputs[0]
                    outputs = torch.softmax(outputs, dim=1)

                    for idx in range(input_ids.shape[0]):
                        probabilities = {}
                        probabilities['id'] = sentence_id[idx]
                        #probability of the second sentence to be "next" to the first one, [idx, 1] corresponds to the positive class
                        probabilities['score'] = outputs[idx, 1].item()

                        loss = self.loss_fn(torch.tensor(outputs[idx, 1].item()), torch.tensor(sentence_label[idx]))
                        probabilities['loss'] = loss.item()

                        predictions.append(probabilities)

        return predictions

    def evaluate(self, intra_targetModel=None, inter_targetModel=None, useTrainingSet=False):
        predictions = {}
        if not self.SKIP_INTERSENTENCE:
            intersentence_predictions = self.evaluate_intersentence(intra_targetModel, useTrainingSet=useTrainingSet)
            predictions['intersentence'] = intersentence_predictions

        if not self.SKIP_INTRASENTENCE:
            intrasentence_predictions = self.evaluate_intrasentence(intra_targetModel, useTrainingSet=useTrainingSet)
            predictions['intrasentence'] = intrasentence_predictions
        return predictions


def process_job(batch, model, pretrained_class):
    input_ids, token_type_ids, sentence_id = batch
    outputs = model(input_ids, token_type_ids=token_type_ids)
    if type(outputs) == tuple:
        outputs = outputs[0]
    outputs = torch.softmax(outputs, dim=1)

    pid = sentence_id[0]
    #probability of the second sentence to be "next" to the first one, [idx, 1] corresponds to the positive class
    pscore = outputs[0, 1].item()
    return (pid, pscore)