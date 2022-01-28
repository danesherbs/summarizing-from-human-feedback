import comet_ml
import torch
import json
import linecache
import random
import json
import transformers
import einops
import os
from tqdm import tqdm

DEVICE = "cuda:2"


def create_tldr_dataset(n_samples=100_000):
    linecount = 0
    with open("tldr/raw_tldr.jsonl", "r") as raw_file:
        with open("tldr/cleaned_tldr.jsonl", "w") as cleaned_file:
            for json_str in tqdm(raw_file):
                post = json.loads(json_str)
                if 24 < post["summary_len"] < 48:
                    cleaned_file.write(json_str)
                    linecount += 1

    assert n_samples <= linecount

    random_idxs = random.sample(range(linecount), k=n_samples)
    with open("tldr/tldr.jsonl", "w") as file:
        for i in tqdm(range(n_samples)):
            line = linecache.getline("tldr/cleaned_tldr.jsonl", lineno=random_idxs[i])
            file.write(line)


class TLDRDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_tldr_dataset):
        self.fname = path_to_tldr_dataset
        self.len = None

    def __len__(self):
        if self.len is None:
            with open(self.fname) as f:
                self.len = sum(1 for line in f)
        return self.len

    def __getitem__(self, i):
        i = i % len(self)
        if not (0 <= i < len(self)):
            raise IndexError(
                f"Tried to retrieve sample at index {i}, but only indicies between 0 and {len(self)-1} modulo {len(self)} are valid."
            )
        line = linecache.getline(self.fname, lineno=i + 1)
        post = json.loads(line)
        return post["normalizedBody"]


def generate_sample(model):
    generate_length = 10
    prompt = "Brand new to Ableton. Loving it so far as I came from a bit so user friendly daw before this. My issue is: when I record a track with no record quantization on... and then choose not to quantize after to give it that swung/human feel... how do I quantize another midi track to the first tracks 'off the grid' timing. ? \n Because the first track (the bass line) is what I formed the section I'm working on (and have been building up from), if I quantize everything else to a super high value grid, that will obviously get them close. But the track will sound slightly messier if everything else remains on the grid (and off from the 'natural' timing of the first bass track). \n TLDR"

    input_ids = tokenizer(
        [prompt],
        max_length=256,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(DEVICE)

    response_ids = model.generate(
        input_ids,
        min_length=input_ids.shape[-1] + generate_length,
        max_length=input_ids.shape[-1] + 10 * generate_length,
        do_sample=True,
        temperature=0.8,
        top_k=len(tokenizer),
        top_p=1.0,
    )

    [decoded] = tokenizer.batch_decode(response_ids)

    return decoded


def collate_fn(batch):
    return tokenizer(
        batch, max_length=256, padding="longest", truncation=True, return_tensors="pt"
    )


def train(model, train_data_loader, epochs=1, lr=1e-3, comet_experiment=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=20
    )

    for _ in range(epochs):
        for inputs in train_data_loader:
            optimizer.zero_grad()

            #  I    am    a    dog    [EOS]    <--- original
            #  I    am    a    dog             <--- inputs (shifted internally)
            #  am   a     dog  [EOS]           <--- targets (shifted internally)
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)
            loss = model(
                input_ids, attention_mask=attention_mask, labels=input_ids
            ).loss
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if comet_experiment is not None:
                comet_experiment.log_metric("train loss", float(loss))
                experiment.log_metric("lr", optimizer.param_groups[0]["lr"])
                experiment.log_text(generate_sample(model))

    if comet_experiment is not None:
        comet_experiment.end()


if __name__ == "__main__":
    dataset = TLDRDataset(path_to_tldr_dataset="tldr/tldr.jsonl")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = -100

    model = transformers.GPT2LMHeadModel.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    ).to(DEVICE)

    data_loader_config = {
        "batch_size": 16,
        "shuffle": True,
        "collate_fn": collate_fn,
    }

    num_train = int(0.95 * len(dataset))
    num_val = len(dataset) - num_train

    data_train, data_val = torch.utils.data.random_split(dataset, (num_train, num_val))
    train_data_loader = torch.utils.data.DataLoader(data_train, **data_loader_config)
    val_data_loader = torch.utils.data.DataLoader(data_val, **data_loader_config)

    experiment = comet_ml.Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="learning-to-summarise-using-human-feedback",
        workspace="danesherbs",
        log_env_cpu=False,
        log_env_gpu=False,
    )

    train(model, train_data_loader, comet_experiment=experiment, lr=3e-5)
