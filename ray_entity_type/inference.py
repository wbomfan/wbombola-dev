import os

import pandas as pd
import torch
from transformers import BertTokenizer

LABELS = [
    "Actor", "Audio", "Book", "Chapter", "Character", "Comic", "Concept", "Event", "Game Mechanic",
    "Location", "Movie", "Object", "Organization", "Portal", "Quest", "Staff", "TVEpisode", "TVSeason",
    "TVSeries", "Trait", "Videogame",
]
TAG_KEY = "entityType"
MAX_LEN = 512
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")


class EntityTypePredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(
            os.path.join(MODEL_DIR, "data/model.pth"), map_location=self.device, weights_only=False
        ).eval()
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_DIR, "tokenizer"))

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        enc = self.tokenizer(
            batch["plaintext_content"].tolist(),
            max_length=MAX_LEN,
            padding=True,
            truncation=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        probs = torch.softmax(logits.float(), dim=1).cpu()
        conf, idx = probs.max(dim=1)
        return pd.DataFrame(
            {
                "wiki_id": batch["wiki_id"].values,
                "article_id": batch["article_id"].values,
                "tag_key": TAG_KEY,
                "tag_value": [LABELS[i] for i in idx.tolist()],
                "confidence_score": conf.numpy(),
            }
        )
