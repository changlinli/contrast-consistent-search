from enum import Enum
from typing import Any

from torch import tensor, Tensor
from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM, \
    PreTrainedTokenizerBase, TensorType, PreTrainedModel
from sklearn.linear_model import LogisticRegression



def get_hidden_state_of_last_layer_last_token(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        input_text: str
) -> Tensor:
    tokens = tokenizer(input_text + tokenizer.eos_token,
                       return_tensors=TensorType.PYTORCH).input_ids.to(model.device)

    with torch.no_grad():
        output = model(tokens, output_hidden_states=True)

    representation_across_all_tuples = output["hidden_states"]
    # Last layer
    layer_to_retrieve = -1
    # Last token
    token_idx_to_retrieve = -1
    representation = representation_across_all_tuples[layer_to_retrieve][0, token_idx_to_retrieve].detach()
    return representation


# Remember whatever we use has to be binary
class Answer(Enum):
    TRUE = 1
    FALSE = 2


def review_opinion_as_text(review_opinion: Answer) -> str:
    match review_opinion:
        case Answer.TRUE:
            return "True"
        case Answer.FALSE:
            return "False"


def int_label_to_answer(label: int) -> Answer:
    match label:
        case 0:
            return Answer.FALSE
        case 1:
            return Answer.TRUE
        case _:
            raise Exception("Got an unexpected label!")


def create_prompt(text: str, label: Answer) -> str:
    """
    Given a review example ("text") and corresponding label (0 for negative, or 1
    for positive), returns a zero-shot prompt for that example (which includes
    that label as the answer).

    (This is just one example of a simple, manually created prompt.)
    """
    return "Give a one word answer of \"True\" or \"False\". Does the following movie review express a positive sentiment?\n" + \
        text + "\n" + review_opinion_as_text(label)


def get_hidden_states_multiple(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, data: Dataset, max_n: int):
    # Make sure that the model is in evaluation mode, not training mode
    model.eval()
    negative_opinions = []
    positive_opinions = []
    ground_truth_labels = []
    for idx_to_retrieve in range(max_n):
        movie_review = data[idx_to_retrieve]["content"]
        ground_truth: int = data[idx_to_retrieve]["label"]
        ground_truth_labels.append(ground_truth)
        positive_opinion = get_hidden_state_of_last_layer_last_token(
            model,
            tokenizer,
            create_prompt(movie_review, Answer.TRUE),
        )
        negative_opinion = get_hidden_state_of_last_layer_last_token(
            model,
            tokenizer,
            create_prompt(movie_review, Answer.FALSE),
        )
        positive_opinions.append(positive_opinion)
        negative_opinions.append(negative_opinion)

    return np.stack(negative_opinions), np.stack(positive_opinions), np.stack(ground_truth_labels)


class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)


class CCS(object):
    def __init__(self, x0, x1, nepochs=1000, ntries=10, lr=1e-3, batch_size=-1,
                 verbose=False, device="cuda", linear=True, weight_decay=0.01, var_normalize=False):
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # probe
        self.linear = linear
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)

    def normalize(self, x):
        """
        Mean-normalizes the data x (of shape (n, d))
        If self.var_normalize, also divides by the standard deviation
        """
        normalized_x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            normalized_x /= normalized_x.std(axis=0, keepdims=True)

        return normalized_x

    def get_tensor_data(self):
        """
        Returns x0, x1 as appropriate tensors (rather than np arrays)
        """
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        return x0, x1

    def get_loss(self, p0, p1):
        """
        Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
        """
        informative_loss = (torch.min(p0, p1) ** 2).mean(0)
        consistent_loss = ((p0 - (1 - p1)) ** 2).mean(0)
        return informative_loss + consistent_loss

    def get_acc(self, x0_test, x1_test, y_test):
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        x0 = torch.tensor(self.normalize(x0_test), dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.normalize(x1_test), dtype=torch.float, requires_grad=False, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5 * (p0 + (1 - p1))
        predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        acc = (predictions == y_test).mean()
        acc = max(acc, 1 - acc)

        return acc

    def train(self):
        """
        Does a single training run of nepochs epochs
        """
        x0, x1 = self.get_tensor_data()
        permutation = torch.randperm(len(x0))
        x0, x1 = x0[permutation], x1[permutation]

        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        # Start training (full batch)
        loss = tensor(0)
        for epoch in range(self.nepochs):
            for j in range(nbatches):
                x0_batch = x0[j * batch_size:(j + 1) * batch_size]
                x1_batch = x1[j * batch_size:(j + 1) * batch_size]

                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()

    def repeated_train(self):
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss

        return best_loss

def main():
    data: Dataset = load_dataset("amazon_polarity")["test"]

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("gpt2-xl")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained("gpt2-xl")

    num_of_examples = 100
    negative_hidden_states, positive_hidden_states, ground_truth_labels = \
        get_hidden_states_multiple(model, tokenizer, data, num_of_examples)
    negative_hidden_states_train = negative_hidden_states[:num_of_examples//2]
    negative_hidden_states_test = negative_hidden_states[num_of_examples//2:]
    positive_hidden_states_train = positive_hidden_states[:num_of_examples//2]
    positive_hidden_states_test = positive_hidden_states[num_of_examples//2:]
    # We don't need a ground truth train because we aren't using ground truth in training!
    ground_truth_test = ground_truth_labels[num_of_examples//2:]
    ccs = CCS(negative_hidden_states_train, positive_hidden_states_train)
    ccs.repeated_train()

    ccs_acc = ccs.get_acc(negative_hidden_states_test, positive_hidden_states_test, ground_truth_labels)
    print("CCS accuracy: {}".format(ccs_acc))

    print("Hello world!")


if __name__ == "__main__":
    main()
