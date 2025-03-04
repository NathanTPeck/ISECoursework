import os

import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from ray import air, train, tune
from ray.tune.search.optuna import OptunaSearch
import optuna
from optuna.samplers import TPESampler

import helper.preprocessing as pp
import helper.data_loader as dl

import copy

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f"Device: {torch.cuda.get_device_name()}")

# Todo 1 Change the below to alter the hyperparameters
LR = 1.3e-3
BATCH_SIZE = 50
DROPOUT = 0.071
MAX_DOC_LEN = 73
TEST_SIZE = 0.2
MAX_VOCAB = 10000
HIDDEN_SIZE = []
POOL_SIZE = 2
FILTER_SIZES = [3, 4]
N_FILTERS = [249, 127]
NUM_EPOCHS = 21
EMBEDDING_TYPE = 1
FREEZE_EMBEDDINGS = False
THRESHOLD = 0.2
# Todo all: Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet')
project = 'keras'
num_iters = 10


class CNN(nn.Module):
    def __init__(self, embed_dim, n_filters, filter_sizes, pool_size, hidden_size, dropout, vocab_size=None,
                 pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()
        if pretrained_embeddings is not None:
            self.vocab_size, self.embed_dim = pretrained_embeddings.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=self.embed_dim, out_channels=n_filters[i], kernel_size=filter_sizes[i])
             for i in range(len(filter_sizes))])
        self.max_pool1 = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        hidden_sizes = [np.sum(n_filters[0:len(filter_sizes)]).item()] + hidden_size + [2]
        self.fcs = nn.ModuleList([
            nn.Linear(in_features=hidden_sizes[i], out_features=hidden_sizes[i + 1], bias=True)
            for i in range(len(hidden_sizes) - 1)
        ])

    def forward(self, text):
        embedded = self.embedding(text).float()
        embedded = embedded.permute(0, 2, 1)
        conv_list = [F.relu(conv(embedded)) for conv in self.convs]
        pool_list = [F.max_pool1d(conv, kernel_size=conv.shape[2]) for conv in conv_list]

        cat = torch.cat([pool.squeeze(dim=2) for pool in pool_list], dim=1)
        x = cat.view(cat.shape[0], -1)

        for i, fc in enumerate(self.fcs):
            if i < len(self.fcs) - 1:
                x = fc(self.relu(x))
                if i < len(self.fcs) - 2:
                    x = self.dropout(x)
            else:
                x = self.dropout(x)
                x = fc(x)
        return x


def initilize_model(
        embed_dim,
        filter_sizes,
        n_filters,
        dropout,
        learning_rate,
        pool_size,
        hidden_size,
        pretrained_embedding=None,
        freeze_embedding=False,
        vocab_size=None
):
    cnn_model = CNN(embed_dim,
                    n_filters,
                    filter_sizes,
                    pool_size,
                    hidden_size,
                    dropout,
                    vocab_size,
                    pretrained_embedding,
                    freeze_embedding)

    cnn_model.to(device)

    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, betas=(0.975, 0.999))

    return cnn_model, optimizer


def train_model(model, optimiser, train_dataloader, test_dataloader=None, epochs=10, threshold=None, verbose=False,
                last=True, best_accuracy=0, best_model_state=None):
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    t0 = time.time()

    if verbose:
        print("Start training...\n")
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Test Loss':^10} | {'Test Acc':^9} | {'Elapsed':^9}")
        print("-" * 60)

    for epoch_i in range(epochs):
        t0_epoch = time.time()
        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(b_input_ids)

            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()

            loss.backward()

            optimiser.step()

        avg_train_loss = total_loss / len(train_dataloader)

        if test_dataloader is not None:
            # Evaluating epoch performance
            test_loss, test_accuracy, y_preds, y_true, y_probs = evaluate(model, test_dataloader, threshold=threshold)

            # Tracking the most accurate model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_model_state = copy.deepcopy(model.state_dict())

            # Printing performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            if verbose:
                print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {test_loss: ^ 10.6f} | {test_accuracy: ^ 9.2f} |"
                      f" {time_elapsed: ^ 9.2f}")

    T = time.time() - t0
    if verbose:
        print(f"total time taken: {T}")

    if best_model_state is not None and last:
        # Loading best model from training epochs
        model.load_state_dict(best_model_state)

    if test_dataloader is not None:
        _, accuracy, y_preds, y_true, y_probs = evaluate(model, test_dataloader, threshold=threshold)

        # Todo 1: change average= to macro for mean average across both classes
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='binary', zero_division=0)
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        auc_score = auc(fpr, tpr)
        if verbose:
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='blue', lw=2,
                     label=f'ROC curve (AUC = {auc_score: .2f})\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}')
            plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            plt.show()

    return accuracy, precision, recall, f1, auc_score, best_model_state


def evaluate(model, test_dataloader, threshold=None):
    model.eval()

    # Tracking variables
    test_accuracy = []
    test_loss = []
    y_probs = []
    y_preds = []
    y_true = []

    # For each batch in our validation set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)
        y_pred_probs = torch.sigmoid(logits)[:, 1]

        # Computing loss
        loss = loss_fn(logits, b_labels)
        test_loss.append(loss.item())

        # Getting the predictions depending on the threshold
        if threshold is None:
            preds = torch.argmax(logits, dim=1).flatten()
        else:
            probs = torch.softmax(logits, dim=1)[:, 1]  # Get probability of class 1
            preds = (probs > threshold).long()

        # Calculating the accuracy rate
        y_probs.extend(y_pred_probs.cpu().numpy())
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        test_accuracy.append(accuracy)
        y_preds.extend(preds.cpu().numpy())
        y_true.extend(b_labels.cpu().numpy())

    # Compute the average accuracy and loss over the validation set.
    test_loss = np.mean(test_loss)
    test_accuracy = accuracy_score(y_true, y_preds)

    return test_loss, test_accuracy, y_preds, y_true, y_probs


def classify_input(model, text, word2idx, max_len):
    model.eval()


    text = pp.clean_str(text)
    # Todo 2: uncommment below to enable stopword removal in the custom input classification
    # text = pp.remove_stopwords(text)

    # Todo 2: Set preprocess to "stem" or "lemmatize" or leave as "" for per word preprocessing
    tokenized_text, _, _ = pp.tokenize_words([text], preprocess="")
    input = pp.pad_and_encode(tokenized_text, word2idx, min(max_len, MAX_DOC_LEN))

    input_tensor = torch.tensor(input, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)

    probability = torch.softmax(logits, dim=1)
    if THRESHOLD is None:
        prediction = torch.argmax(probability, dim=1).item()
    else:
        prediction = (probability[:, 1] > THRESHOLD).long().item()

    prob_tuple = tuple(probability.cpu().numpy()[0])

    return prediction, prob_tuple


print("Processing dataset headers...")
pp.process_dataset(project)

data = pd.read_csv("Title+Body.csv").fillna("")
text_col = "text"
labels = data["sentiment"].values

print("Preprocessing data...")
# Todo 1: comment out the line below to remove string cleaning (also used for lowercasing, but not enabled by default)
data[text_col] = data[text_col].apply(pp.clean_str)
# Todo 1: uncomment the 2 lines below to enable stopword removal
# nltk.download('stopwords', quiet=True)
# data[text_col] = data[text_col].apply(pp.remove_stopwords)

print("Tokenizing words...")
# Todo 1: Set preprocess to "stem" or "lemmatize" or leave as "" for per word preprocessing
tokenized_texts, word2idx, max_len = pp.tokenize_words(data[text_col], preprocess="")

input_ids = pp.pad_and_encode(tokenized_texts, word2idx, min(max_len, MAX_DOC_LEN))

embeddings = pp.load_pretrained_vectors(word2idx, "embeddings/crawl-300d-2M.vec")
embeddings = torch.tensor(embeddings, dtype=torch.float32)
EMBEDDING_DIM = 300


initial_embeddings = embeddings
loss_fn = nn.CrossEntropyLoss()


def benchmark_models():
    out_csv_name = f'./results/{project}_CNN.csv'

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []

    for i in range(num_iters):
        X_train, X_test, y_train, y_test = train_test_split(input_ids, labels, test_size=TEST_SIZE, random_state=42)
        train_dataloader, test_dataloader = dl.data_loader(X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE)

        embeddings = initial_embeddings
        cnn_model, optimiser = initilize_model(EMBEDDING_DIM, FILTER_SIZES, N_FILTERS, DROPOUT, LR, POOL_SIZE,
                                               HIDDEN_SIZE, pretrained_embedding=embeddings, vocab_size=MAX_VOCAB,
                                               freeze_embedding=FREEZE_EMBEDDINGS)

        acc, pre, rec, xf1, rocauc, _ = train_model(cnn_model, optimiser, train_dataloader, test_dataloader,
                                                    epochs=NUM_EPOCHS, threshold=THRESHOLD, verbose=True)
        accuracies.append(acc)
        precisions.append(pre)
        recalls.append(rec)
        f1s.append(xf1)
        roc_aucs.append(rocauc)

    final_accuracy = np.mean(accuracies)
    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    final_f1 = np.mean(f1s)
    final_auc = np.mean(roc_aucs)

    print("\n")
    print("=== CNN Results ===")
    print(f"Accuracy:   {final_accuracy:.4f}")
    print(f"Precision:  {final_precision:.4f}")
    print(f"Recall:     {final_recall:.4f}")
    print(f"F1-Score:   {final_f1:.4f}")
    print(f"AUC:        {final_auc:.4f}")

    print(f"proposed_results_array = {[accuracies, precisions, recalls, f1s, roc_aucs]}")

    header_needed = not os.path.isfile(out_csv_name)

    df_log = pd.DataFrame(
        {
            'repeated_times': [num_iters],
            'Accuracy': [final_accuracy],
            'Precision': [final_precision],
            'Recall': [final_recall],
            'F1': [final_f1],
            'AUC': [final_auc],
        }
    )

    df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)
    print(f"\nResults have been saved to: {out_csv_name}")
    return cnn_model


def objective(config):
    X_train, X_test, y_train, y_test = train_test_split(input_ids, labels, test_size=TEST_SIZE, random_state=42)
    train_dataloader, test_dataloader = dl.data_loader(X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE)

    embeddings = initial_embeddings

    best_model_state = None
    best_accuracy = 0

    model, optimiser = initilize_model(
        EMBEDDING_DIM,
        config["filter_sizes"],
        [config["num_filters_1"], config["num_filters_2"]],
        config["dropout"],
        config["lr"],
        POOL_SIZE,
        config["hidden_size"],
        pretrained_embedding=embeddings, vocab_size=MAX_VOCAB, freeze_embedding=FREEZE_EMBEDDINGS)

    for i in range(config["epochs"]):
        last = (i == (config["epochs"] - 1))
        acc, _, _, f1, _, best_model_state = train_model(model, optimiser, train_dataloader,
                                                         test_dataloader=test_dataloader, epochs=1,
                                                         threshold=config["threshold"], last=last,
                                                         best_accuracy=best_accuracy, best_model_state=best_model_state)
        best_accuracy = max(acc, best_accuracy)
        train.report({"f1": f1, "accuracy": acc})


def short_dirname(trial):
    return "trial_" + str(trial.trial_id)


def define_search_space(trial: optuna.Trial):
    # Todo 3: change the following hyperparameters as you like
    trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    trial.suggest_float("dropout", 0.05, 0.5)
    trial.suggest_categorical("filter_sizes", [[2, 3], [2, 4], [3, 4]])
    trial.suggest_int("num_filters_1", 64, 512)
    trial.suggest_int("num_filters_2", 64, 512)
    trial.suggest_int("epochs", 10, 40)
    trial.suggest_categorical("hidden_size", [[], [256], [128], [64], [256, 128], [256, 64], [128, 64]])
    trial.suggest_float("threshold", 0.05, 0.5)


def tune_hyperparameters():
    algo = OptunaSearch(define_search_space, metric="accuracy", mode="max", sampler=TPESampler())

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        # Todo 3: Ensure max_t is equal to the max epochs tested
        max_t=40,
        grace_period=10,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            trial_dirname_creator=short_dirname,
            # Todo 3: adjust to computational potential
            num_samples=5,
            search_alg=algo,
            scheduler=scheduler
        ),
        run_config=train.RunConfig(
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=0),
            # Todo 3: adjust to wherever you like
            storage_path="ISECoursework\\results\\RayTuner",
        ),
    )

    results = tuner.fit()
    print("Best config is:", results.get_best_result(metric="accuracy", mode="max").config)


# Todo 1: Comment out if tuning hyperparameters
cnn_model = benchmark_models()

# Todo 3: Uncomment to tune hyperparameters
# tune_hyperparameters()


def user_input_prediction(model):
    while True:
        user_input = input("Enter a bug report to classify:\n")
        if user_input == "quit":
            break
        label, probabilities = classify_input(model, user_input, word2idx, max_len)
        print(f"Predicted Label: {label}")
        print(f"Probabilities: {probabilities}")


# Todo 2: uncomment below code to test your own text input for the model to classify
#  (note: this is unstable with very small inputs)
# user_input_prediction(cnn_model)
