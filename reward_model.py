import comet_ml
import torch
import json
import linecache
import collections
import transformers
import os


DEVICE = "cuda:3"


class ComparisonDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_dataset_dir):
        self.path_to_dataset_dir = path_to_dataset_dir
        self.file_names = [f"{path_to_dataset_dir}/batch{i}.json" for i in range(3, 11)]
        self.file_lengths = None

    def __len__(self):
        if self.file_lengths is None:
            self.file_lengths = collections.OrderedDict()
            for file_name in self.file_names:
                with open(file_name) as f:
                    self.file_lengths[file_name] = sum(1 for line in f)

        return sum(self.file_lengths.values())

    def __getitem__(self, i):
        i = i % len(self)

        if not (0 <= i < len(self)):
            raise IndexError(
                f"Tried to retrieve sample at index {i}, but only indicies between 0 and {len(self)-1} modulo {len(self)} are valid."
            )

        cum_length = 0

        for file_name in self.file_names:
            cum_length += self.file_lengths[file_name]
            if i < cum_length:
                file_idx = i - cum_length + self.file_lengths[file_name]
                line = linecache.getline(file_name, lineno=file_idx + 1)
                payload = json.loads(line)
                choice = payload["choice"]
                summary_good = payload["summaries"][choice]["text"]
                summary_bad = payload["summaries"][1 - choice]["text"]
                post = payload["info"]["post"]
                post_good = f"{post} TLDR:{summary_good}"
                post_bad = f"{post} TLDR:{summary_bad}"
                return post_good, post_bad


class GPTWithRewardHead(torch.nn.Module):
    def __init__(self, mask_token_id=-100):
        super().__init__()
        self.gpt = transformers.GPT2LMHeadModel.from_pretrained(
            "gpt2", pad_token_id=mask_token_id
        )
        self.generate = self.gpt.generate  # borrow existing generate function
        hidden_size = self.gpt.transformer.wte.weight.shape[-1]
        self.reward_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 4 * hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * hidden_size, 1),
        )

    def forward(self, input_ids, **kwargs):
        response = self.gpt(
            input_ids, output_hidden_states=True, **kwargs
        )  # [batch_size, num_layers, hidden_dim]
        last_hidden_state = response.hidden_states[
            -1
        ]  # [batch_size, seq_len, hidden_size]
        rewards = self.reward_network(last_hidden_state).squeeze(-1)
        last_reward = rewards[:, -1]
        logits = response.logits  # [batch_size, seq_len, vocab_size]
        return logits, last_reward


def collate_fn(batches):
    summary_good, summary_bad = zip(*batches)
    tokens_good = tokenizer(summary_good, **tokenizer_config)
    tokens_bad = tokenizer(summary_bad, **tokenizer_config)
    return tokens_good, tokens_bad


def train(model, train_data_loader, epochs=30, lr=1e-3, comet_experiment=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=20
    )

    for _ in range(epochs):
        for inputs_good, inputs_bad in train_data_loader:
            optimizer.zero_grad()

            input_good_ids = inputs_good["input_ids"].to(DEVICE)
            attention_good_mask = inputs_good["attention_mask"].to(DEVICE)
            _, rewards_good = model(input_good_ids, attention_mask=attention_good_mask)

            input_bad_ids = inputs_bad["input_ids"].to(DEVICE)
            attention_bad_mask = inputs_bad["attention_mask"].to(DEVICE)
            _, rewards_bad = model(input_bad_ids, attention_mask=attention_bad_mask)

            loss = torch.log(torch.sigmoid(rewards_good - rewards_bad)).mean()

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if comet_experiment is not None:
                comet_experiment.log_metric("train loss", float(loss))
                experiment.log_metric("lr", optimizer.param_groups[0]["lr"])

    if comet_experiment is not None:
        comet_experiment.end()


if __name__ == "__main__":
    dataset = ComparisonDataset(path_to_dataset_dir="./comparisons")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = -100
    model = GPTWithRewardHead().to(DEVICE)

    tokenizer_config = {
        "max_length": 512,
        "padding": "longest",
        "truncation": True,
        "return_tensors": "pt",
    }

    data_loader_config = {
        "batch_size": 4,
        "shuffle": True,
        "collate_fn": collate_fn,
    }

    num_train = int(0.95 * len(dataset))
    num_test = len(dataset) - num_train
    data_train, data_test = torch.utils.data.random_split(
        dataset, (num_train, num_test)
    )
    train_data_loader = torch.utils.data.DataLoader(data_train, **data_loader_config)
    test_data_loader = torch.utils.data.DataLoader(data_test, **data_loader_config)

    experiment = comet_ml.Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="learning-to-summarise-using-human-feedback",
        workspace="danesherbs",
        log_env_cpu=False,
        log_env_gpu=False,
    )

    train(model, train_data_loader, comet_experiment=experiment, lr=3e-5)
