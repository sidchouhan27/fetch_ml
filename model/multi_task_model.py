import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=3, num_ner_labels=5):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        self.ner_classifier = nn.Linear(hidden_size, num_ner_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]  # sentence-level

        logits_task_a = self.classifier(cls_embedding)
        logits_task_b = self.ner_classifier(last_hidden_state)
        return logits_task_a, logits_task_b