"""
Model classes - required by PyTorch to load model
"""
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BertClassifierV1(nn.Module):  # pylint: disable=missing-class-docstring
    PRETRAINED_NAME = "bert-base-cased"

    def __init__(self, labels, dropout=0.5):
        super().__init__()
        self.labels = labels
        self.labels_reverse = {v: k for k, v in self.labels.items()}

        self.bert = AutoModel.from_pretrained(self.PRETRAINED_NAME)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 21)
        self.relu = nn.ReLU()

        self.max_input_tokens = self.bert.config.max_position_embeddings

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


class BertClassifierV2(nn.Module):  # pylint: disable=missing-class-docstring
    PRETRAINED_NAME = "bert-base-cased"

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.labels_reverse = {v: k for k, v in self.labels.items()}

        self.bert = AutoModel.from_pretrained(self.PRETRAINED_NAME)

        embedding_dim_size = self.bert.config.to_dict()["hidden_size"]
        self.linear = nn.Linear(embedding_dim_size, len(self.labels))

        self.max_input_tokens = self.bert.config.max_position_embeddings

    def forward(self, input_ids, attention_mask):
        # outputs[0] includes bert output for each token, outputs[1] - pooled output
        outputs = self.bert(input_ids, attention_mask=attention_mask)[1]
        # clf head
        outputs = self.linear(outputs)
        return outputs


class DistilBERTClassifier(nn.Module):  # pylint: disable=missing-class-docstring
    PRETRAINED_NAME = "distilbert-base-cased"

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.labels_reverse = {v: k for k, v in self.labels.items()}

        self.base = AutoModel.from_pretrained(self.PRETRAINED_NAME)
        embedding_dim_size = self.base.config.dim
        self.linear = nn.Linear(embedding_dim_size, len(self.labels))

        self.max_input_tokens = self.base.config.max_position_embeddings

    def forward(self, input_ids, attention_mask):
        # pooled output - embedding for first token
        outputs = self.base(input_ids, attention_mask=attention_mask)[0][:, 0]
        # clf head
        outputs = self.linear(outputs)
        return outputs


class RoBERTaClassifier(nn.Module):  # pylint: disable=missing-class-docstring
    PRETRAINED_NAME = "roberta-base"

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.labels_reverse = {v: k for k, v in self.labels.items()}

        self.base = AutoModel.from_pretrained(self.PRETRAINED_NAME)
        embedding_dim_size = self.base.config.hidden_size
        self.linear = nn.Linear(embedding_dim_size, len(self.labels))

        self.max_input_tokens = 512  # self.base.config.max_position_embeddings

    def forward(self, input_ids, attention_mask):
        # outputs[0] includes bert output for each token, outputs[1] - pooled output
        outputs = self.base(input_ids, attention_mask=attention_mask)[1]
        # clf head
        outputs = self.linear(outputs)
        return outputs


def get_model_class_and_tokenizer(model_name):
    model_classes = {
        "bert1": BertClassifierV1,
        "bert2": BertClassifierV2,
        "distilBERT": DistilBERTClassifier,
        "RoBERTa": RoBERTaClassifier,
    }
    select_class = model_classes[model_name]
    return model_classes[model_name], AutoTokenizer.from_pretrained(select_class.PRETRAINED_NAME)
