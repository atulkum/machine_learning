import numpy as np

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

import os
import random
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

bert_config = BertConfig.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

is_cuda = torch.cuda.is_available()

label_list = [
    "neutral",
    "entailment",
    "contradiction"
]

label_map = {
    "neutral":0,
    "entailment":1,
    "contradiction":2
}
def load_examples(filename, is_train=True):
    all_input_ids1 = []
    all_input_ids2 = []
    all_labels = []
    for i, line in enumerate(open(filename)):
        p = line.split("\t")
        all_input_ids1.append(torch.tensor(bert_tokenizer.encode(p[1])).long())
        all_input_ids2.append(torch.tensor(bert_tokenizer.encode(p[2])).long())

        if is_train:
            all_labels.append(label_map[p[0]])

    all_input_ids1 = pad_sequence(all_input_ids1, batch_first=True, padding_value=bert_tokenizer.pad_token_id)
    all_input_ids2 = pad_sequence(all_input_ids2, batch_first=True, padding_value=bert_tokenizer.pad_token_id)

    if is_train:
        all_labels = torch.tensor(all_labels).long()
        dataset = TensorDataset(all_input_ids1, all_input_ids2, all_labels)
    else:
        dataset = TensorDataset(all_input_ids1, all_input_ids2)
    return dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed_all(seed)


class BertNLI(nn.Module):
    def __init__(self):
        super(BertNLI, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        proj_dim = 64
        self.embd_proj = nn.Linear(bert_config.hidden_size, proj_dim)
        self.bilinear_proj = nn.Bilinear(proj_dim, proj_dim, proj_dim, bias=False)
        self.classifier = nn.Linear(proj_dim, 3)

        self.embd_proj.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)
        self.embd_proj.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)
        self.classifier.bias.data.zero_()
        self.bilinear_proj.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)

    def forward(self, input_ids1, input_ids2, labels = False):
        attention_mask = input_ids1.ne(bert_tokenizer.pad_token_id)
        outputs = self.bert(
            input_ids=input_ids1,
            attention_mask=attention_mask
        )
        class_output1 = self.dropout(outputs[1])
        class_output1 = self.embd_proj(class_output1)

        attention_mask = input_ids2.ne(bert_tokenizer.pad_token_id)
        outputs = self.bert(
            input_ids=input_ids2,
            attention_mask=attention_mask
        )
        class_output2 = self.dropout(outputs[1])
        class_output2 = self.embd_proj(class_output2)

        proj = self.bilinear_proj(class_output1, class_output2)
        logits = self.classifier(proj)
        logits = F.log_softmax(logits, dim=1)

        if self.training:
            return F.nll_loss(logits, labels, reduction='mean')
        else:
            score, pred = logits.max(dim=1)
            score = score.exp()
            return score, pred

def evaluation(model, eval_dataset):
    y = None
    gd = None
    epoch_iterator = tqdm(eval_dataset, desc="Evaluation", disable=False)
    for step, batch in enumerate(epoch_iterator):
        model.eval()
        with torch.no_grad():
            if is_cuda:
                batch = tuple(t.cuda() for t in batch)

            inputs = {"input_ids1": batch[0], "input_ids2": batch[1], "labels": batch[2]}
            _, p = model(**inputs)
            if y is not None:
                y = np.append(y, p.cpu().data.numpy())
                gd = np.append(gd, batch[2].cpu().data.numpy())
            else:
                y = p.cpu().data.numpy()
                gd = batch[2].cpu().data.numpy()
    return np.mean(y == gd)

def prediction(model, test_dataset, filename):
    y = []
    epoch_iterator = tqdm(test_dataset, desc="Testing", disable=False)
    for step, batch in enumerate(epoch_iterator):
        model.eval()
        if is_cuda:
            batch = tuple(t.cuda() for t in batch)
        with torch.no_grad():
            inputs = {"input_ids1": batch[0], "input_ids2": batch[1]}
            _, p = model(**inputs)
            y.extend(list(p.cpu().data.numpy()))
    with open(filename, 'w') as f:
        for yi in y:
            f.write(label_list[yi])
            f.write('\n')

def train(dataset, output_root_dir, test_filename):
    weight_decay = 0.0
    warmup_steps = 0.0
    learning_rate = 5e-5
    adam_epsilon = 1e-8
    num_train_epochs = 3
    train_batch_size = 32
    seed = 42
    logging_steps = 1000
    max_grad_norm = 1.

    model = BertNLI()
    if is_cuda:
        model = model.cuda()
    #train test split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler)
    eval_dataloader = DataLoader(dataset, batch_size=train_batch_size, sampler=valid_sampler)

    t_total = num_train_epochs * len(train_dataloader)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(0, int(num_train_epochs), desc="Epoch", disable=False)
    set_seed(seed)
    for _ in train_iterator:
        optimizer.zero_grad()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            if is_cuda:
                batch = tuple(t.cuda() for t in batch)

            inputs = {"input_ids1": batch[0], "input_ids2": batch[1], "labels": batch[2]}
            model.train()
            loss = model(**inputs)
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if global_step % logging_steps == 0:
                print(f"{global_step}/{t_total}: lr: {scheduler.get_last_lr()}")
                print(f"{global_step}/{t_total}: loss: {(tr_loss - logging_loss)/logging_steps}")
                logging_loss = tr_loss

        ##eval
        acc = evaluation(model, eval_dataloader)
        print(f"{global_step}: acc: {acc}")

    save_directory = os.path.join(output_root_dir, 'checkpoint')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    print(f"Saving model checkpoint to {save_directory}", flush=True)
    output_model_file = os.path.join(save_directory, "pytorch_model.bin")
    torch.save(model.state_dict(), output_model_file)

    #dump test
    all_test = load_examples(test_filename)
    test_sampler = SequentialSampler(all_test)
    test_dataloader = DataLoader(all_test, sampler=test_sampler, batch_size=train_batch_size)
    prediction(model, test_dataloader, os.path.join(output_root_dir, 'submission.txt'))

if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser("~"), 'Downloads/entailment1')
    test_filename = os.path.join(root_dir, 'test_data.txt')
    train_filename = os.path.join(root_dir, 'train_data.txt')
    output_root_dir = os.path.join(root_dir, f'dl_model_{int(time.time())}')

    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    all_train = load_examples(train_filename)

    train(all_train, output_root_dir, test_filename)
