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
    "spam",
    "ham"
]

label_map = {
    "spam":0,
    "ham":1
}
def load_examples(filename, is_train=True):
    all_input_ids = []
    all_labels = []
    for i, line in enumerate(open(filename)):
        if is_train:
            p = line.split(",")
            all_labels.append(label_map[p[0]])
            rest_str = ','.join(p[1:])
            all_input_ids.append(torch.tensor(bert_tokenizer.encode(rest_str)).long())
        else:
            all_input_ids.append(torch.tensor(bert_tokenizer.encode(line)).long())

    all_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=bert_tokenizer.pad_token_id)

    if is_train:
        all_labels = torch.tensor(all_labels).long()
        dataset = TensorDataset(all_input_ids, all_labels)
    else:
        dataset = TensorDataset(all_input_ids)
    return dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed_all(seed)


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        proj_dim = 64
        self.embd_proj = nn.Linear(bert_config.hidden_size, proj_dim)
        self.classifier = nn.Linear(proj_dim, 2)

        self.embd_proj.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)
        self.embd_proj.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(self, input_ids, labels = None):
        attention_mask = input_ids.ne(bert_tokenizer.pad_token_id)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        class_output = self.dropout(outputs[1])
        proj = self.embd_proj(class_output)
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

            inputs = {"input_ids": batch[0], "labels": batch[1]}
            _, p = model(**inputs)
            if y is not None:
                y = np.append(y, p.cpu().data.numpy())
                gd = np.append(gd, batch[1].cpu().data.numpy())
            else:
                y = p.cpu().data.numpy()
                gd = batch[1].cpu().data.numpy()
    return np.mean(y == gd)

def prediction(model, test_dataset, filename):
    y = []
    epoch_iterator = tqdm(test_dataset, desc="Testing", disable=False)
    for step, batch in enumerate(epoch_iterator):
        model.eval()
        if is_cuda:
            batch = tuple(t.cuda() for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0]}
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
    train_batch_size = 8
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

            inputs = {"input_ids": batch[0], "labels": batch[1]}
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

    all_test = load_examples(test_filename, False)
    test_sampler = SequentialSampler(all_test)
    test_dataloader = DataLoader(all_test, sampler=test_sampler, batch_size=train_batch_size)
    prediction(model, test_dataloader, os.path.join(output_root_dir, 'submission.txt'))

def dump_test(test_filename, model_dir):
    model = BertNLI()
    save_directory = os.path.join(model_dir, 'checkpoint')
    model_file_path = os.path.join(save_directory, "pytorch_model.bin")
    print(f'reading model from {model_file_path}')
    state_dict = torch.load(model_file_path, map_location=lambda storage, location: storage)
    model.eval()
    model.load_state_dict(state_dict, strict=False)
    if is_cuda:
        model = model.cuda()

    #dump test
    batch_size = 32
    all_test = load_examples(test_filename, False)
    test_sampler = SequentialSampler(all_test)
    test_dataloader = DataLoader(all_test, sampler=test_sampler, batch_size=batch_size)
    prediction(model, test_dataloader, os.path.join(model_dir, 'submission.txt'))

if __name__ == "__main__":
    root_dir = os.path.join(os.path.expanduser("~"), 'Downloads/spam1')
    test_filename = os.path.join(root_dir, 'test_data.txt')
    #dump_test(test_filename, os.path.join(root_dir, "dl_model_1592011278"))

    train_filename = os.path.join(root_dir, 'train_data.txt')
    output_root_dir = os.path.join(root_dir, f'dl_model_{int(time.time())}')
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    all_train = load_examples(train_filename)
    train(all_train, output_root_dir, test_filename)
