import copy
from enum import Enum
from typing import Literal

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from sklearn.cluster import KMeans
from torch import tensor, Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    PreTrainedTokenizerBase, TensorType, PreTrainedModel


def generate_text_from_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        max_length: int,
) -> str:
    tokens = tokenizer(prompt, return_tensors=TensorType.PYTORCH).input_ids.to(model.device)
    outputs = model.generate(tokens, max_length=max_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


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
            return "Yes"
        case Answer.FALSE:
            return "No"


def int_label_to_answer(label: int) -> Answer:
    match label:
        case 0:
            return Answer.FALSE
        case 1:
            return Answer.TRUE
        case _:
            raise Exception("Got an unexpected label!")


PROMPT_TEXT = \
    "Give a one word answer of \"Yes\" or \"No\". Do the following reviews express a positive sentiment?\n" + \
    "If you've played the game, you know how divine the music is! Every single song tells a story of the game in the most perfect way possible.\nYour Answer: Yes.\n" + \
    "I guess you have to be a romance novel lover for this one, and not a very discerning one. All others beware! It is absolute drivel.\nYour Answer: No.\n" + \
    "I feel I have to write to keep others from wasting their money. This book seems to have been written by a 7th grader with poor grammatical skills for her age! As another reviewer points out, there is a misspelling on the cover, and I believe there is at least one per chapter. For example, it was mentioned twice that she had a 'lean' on her house. I was so distracted by the poor writing and weak plot, that I decided to read with a pencil in hand to mark all of the horrible grammar and spelling. Please don't waste your money.\nYour Answer: No.\n"


def create_prompt(text: str, label: Answer) -> str:
    """
    Given a review example ("text") and corresponding label (0 for negative, or 1
    for positive), returns a zero-shot prompt for that example (which includes
    that label as the answer).

    (This is just one example of a simple, manually created prompt.)
    """
    return PROMPT_TEXT + text + "\nYour Answer: " + review_opinion_as_text(label)


def get_hidden_states_multiple(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, data: Dataset, max_n: int):
    # Make sure that the model is in evaluation mode, not training mode
    model.eval()
    negative_opinions = []
    positive_opinions = []
    ground_truth_labels = []
    for idx_to_retrieve in tqdm(range(max_n)):
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
    def __init__(
            self,
            x0: numpy.ndarray,
            x1: numpy.ndarray,
            nepochs: int = 1000,
            ntries: int = 10,
            lr: float = 1e-3,
            batch_size: int = -1,
            device: Literal["cpu", "cuda"] = "cuda",
            single_layer=True,
            weight_decay=0.01,
            var_normalize=False,
    ):
        """

        @param x0: Hidden state representations of "positive" statement vectors (it doesn't really have to be positive,
            just needs to be the opposite of x1).
        @param x1: Hidden state representations of "negative" statement vectors (it doesn't really have to be negative,
            just needs to be the opposite of x0).
        @param nepochs: The number of epochs we wish to train our probe's neural net for.
        @param ntries: The number of times we will restart the training process to try to find the best set of weights
            for our probe's neural net.
        @param lr: Learning rate used for the Adam optimizer in gradient descent for the probe's neural net.
        @param batch_size: Size of batches use for training the probe's neural net. Set to -1 if you want the entire
            training set used as a single batch.
        @param device: Which device will store the tensors.
        @param single_layer: Whether we will have a single layer neural net or a multi-layer neural net used as our
            probe.
        @param weight_decay: Weight decay parameter for the Adam optimizer in gradient descent.
        @param var_normalize: Whether to normalize the variance in our training set. Along with mean normalization
            (without which CCS basically just doesn't work, so it's not set as an option), this is used to try to remove
            the effect of the difference in the words themselves.
        """
        # data
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.d = self.x0.shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        # probe
        self.linear = single_layer
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
        predictions: numpy.ndarray = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
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
        for _ in tqdm(range(self.nepochs)):
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


class ClusterTruth(Enum):
    CLUSTER_0_IS_TRUTH = 1
    CLUSTER_1_IS_TRUTH = 2


trivially_positive_review = "This book was amazing! It's the best thing I have ever read in my entire life. I am so " \
                            "happy after reading this."


def kmeans_predict_once(
        kmeans: KMeans,
        cluster_truthiness: ClusterTruth,
        input_vector: numpy.ndarray,
) -> bool:
    result_as_array: numpy.ndarray = kmeans.predict([input_vector])
    result_as_int = result_as_array[0]
    match cluster_truthiness:
        case ClusterTruth.CLUSTER_0_IS_TRUTH:
            return True if result_as_int == 0 else False
        case ClusterTruth.CLUSTER_1_IS_TRUTH:
            return True if result_as_int == 1 else False


def contrastive_pairs_with_k_means_on_text(
        kmeans: KMeans,
        cluster_truthiness: ClusterTruth,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        input_text: str,
) -> bool:
    hidden_state = get_hidden_state_of_last_layer_last_token(model, tokenizer, input_text)
    return kmeans_predict_once(kmeans, cluster_truthiness, hidden_state.numpy())


def evaluate_k_means_accuracy(
        hidden_states_train: numpy.ndarray,
        hidden_states_test: numpy.ndarray,
        ground_truth_labels: numpy.ndarray,
) -> float:
    # We just throw it all together into K-Means and see what comes out
    kmeans = KMeans(n_clusters=2)
    normalized_hidden_states_train = hidden_states_train - hidden_states_train.mean(axis=0)
    kmeans.fit(normalized_hidden_states_train.astype('float'))
    cluster_truthiness = ClusterTruth.CLUSTER_0_IS_TRUTH
    output = [(kmeans_predict_once(kmeans, cluster_truthiness, hidden_state), ground_truth) for
              hidden_state, ground_truth in
              zip(hidden_states_test, ground_truth_labels)]
    accuracy = len(list(filter(lambda x: x[0] == x[1], output))) / len(output)
    print(f"{accuracy=}")
    return accuracy




def main():
    k_means_train_data = torch.rand((100, 50), dtype=torch.float)
    k_means_test_data = torch.rand((200, 50), dtype=torch.float)
    ground_truth_labels = tensor([i < 0.5 for i in torch.rand(200, dtype=torch.float)])
    print(f"{k_means_test_data=}")
    evaluate_k_means_accuracy(
        hidden_states_train=k_means_train_data.numpy(),
        hidden_states_test=k_means_test_data.numpy(),
        ground_truth_labels=ground_truth_labels.numpy(),
    )
    data: Dataset = load_dataset("amazon_polarity")["test"]

    # Can try with GPT-J
    # model_name = "EleutherAI/gpt-j-6B"
    # tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    # You'll probably want to use float16 because otherwise memory usage is huge
    # model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name, revision="float16", torch_dtype=torch.float16)
    model_name = "gpt2-xl"
    print("Beginning to download tokenizer...")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    print("Beginning to download model...")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
    print("Finished downloading model...")
    example_text_prompt = \
        PROMPT_TEXT + "This soundtrack is my favorite music of all time, hands down." + "\n"
    example_text = generate_text_from_model(model, tokenizer, example_text_prompt, 500)
    print(f"Example text generated from the model: {example_text}")

    num_of_examples = 100
    negative_hidden_states, positive_hidden_states, ground_truth_labels = \
        get_hidden_states_multiple(model, tokenizer, data, num_of_examples)
    negative_hidden_states_train = negative_hidden_states[:num_of_examples // 2]
    negative_hidden_states_test = negative_hidden_states[num_of_examples // 2:]
    positive_hidden_states_train = positive_hidden_states[:num_of_examples // 2]
    positive_hidden_states_test = positive_hidden_states[num_of_examples // 2:]
    # We don't need a ground truth train because we aren't using ground truth in training!
    ground_truth_test = ground_truth_labels[num_of_examples // 2:]
    # ccs = CCS(negative_hidden_states_train, positive_hidden_states_train)
    # print("Beginning CCS training...")
    # ccs.repeated_train()
    #
    # ccs_acc = ccs.get_acc(negative_hidden_states_test, positive_hidden_states_test, ground_truth_test)
    # print("CCS accuracy: {}".format(ccs_acc))
    evaluate_k_means_accuracy(
        hidden_states_test=negative_hidden_states_test + positive_hidden_states_test,
        hidden_states_train=negative_hidden_states_train + positive_hidden_states_train,
        ground_truth_labels=ground_truth_test
    )

    print("Hello world!")


if __name__ == "__main__":
    main()
