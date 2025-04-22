import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from model.multi_task_model import MultiTaskSentenceTransformer

class DummyDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100):
        self.samples = [f"Sample sentence {i}" for i in range(num_samples)]
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.samples[idx], padding='max_length', truncation=True,
                                max_length=16, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in tokens.items()}
        item['label_class'] = torch.tensor(idx % 3)
        item['label_ner'] = torch.randint(0, 5, (16,))
        return item

    def __len__(self): return len(self.samples)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = MultiTaskSentenceTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_class = nn.CrossEntropyLoss()
loss_ner = nn.CrossEntropyLoss()
loader = DataLoader(DummyDataset(tokenizer), batch_size=8)

for epoch in range(2):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        a_logits, b_logits = model(batch['input_ids'], batch['attention_mask'])
        loss_a = loss_class(a_logits, batch['label_class'])
        loss_b = loss_ner(b_logits.view(-1, b_logits.shape[-1]), batch['label_ner'].view(-1))
        total_loss = 0.5 * loss_a + 0.5 * loss_b
        total_loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} | LossA: {loss_a.item():.4f} | LossB: {loss_b.item():.4f}")