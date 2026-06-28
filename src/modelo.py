import gc
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


class EstrategiaEmbedding(Enum):
    MEDIA_GLOBAL = "mean"
    PRIMEROS_K = "first_k"
    ULTIMOS_K = "last_k"


class EmbeddingExtractor:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_name)
        self.embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        full = AutoModelForCausalLM.from_pretrained(model_name)
        self.embed.weight.data.copy_(full.model.embed_tokens.weight.data)
        del full
        gc.collect()
        self.embed.to(self.device).eval()

    def _embedding_mean(self, hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        emb = (hs * mask_f).sum(dim=1) / mask_f.sum(dim=1)
        return emb

    def _embedding_first_k(self, hs: torch.Tensor, mask: torch.Tensor, k: int = 3) -> torch.Tensor:
        embs = []
        for b in range(hs.shape[0]):
            seq_len = mask[b].sum().int().item()
            end = min(k, seq_len)
            embs.append(hs[b, :end].mean(dim=0))
        return torch.stack(embs)

    def _embedding_last_k(self, hs: torch.Tensor, mask: torch.Tensor, k: int = 3) -> torch.Tensor:
        embs = []
        for b in range(hs.shape[0]):
            seq_len = mask[b].sum().int().item()
            start = max(0, seq_len - k)
            embs.append(hs[b, start:seq_len].mean(dim=0))
        return torch.stack(embs)

    def extraer_embedding(
        self,
        texto: str,
        estrategia: EstrategiaEmbedding = EstrategiaEmbedding.MEDIA_GLOBAL,
    ) -> np.ndarray:
        inputs = self.tokenizer(texto, return_tensors="pt").to(self.device)
        hs = self.embed(inputs.input_ids).detach().float()
        mask = inputs.attention_mask

        if estrategia == EstrategiaEmbedding.MEDIA_GLOBAL:
            emb = self._embedding_mean(hs, mask)
        elif estrategia == EstrategiaEmbedding.PRIMEROS_K:
            emb = self._embedding_first_k(hs, mask)
        elif estrategia == EstrategiaEmbedding.ULTIMOS_K:
            emb = self._embedding_last_k(hs, mask)

        emb = F.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().numpy().astype(np.float16)

    def extraer_embeddings(
        self,
        textos: list[str],
        estrategia: EstrategiaEmbedding = EstrategiaEmbedding.MEDIA_GLOBAL,
    ) -> np.ndarray:
        inputs = self.tokenizer(textos, return_tensors="pt", padding=True).to(self.device)
        hs = self.embed(inputs.input_ids).detach().float()
        mask = inputs.attention_mask

        if estrategia == EstrategiaEmbedding.MEDIA_GLOBAL:
            emb = self._embedding_mean(hs, mask)
        elif estrategia == EstrategiaEmbedding.PRIMEROS_K:
            emb = self._embedding_first_k(hs, mask)
        elif estrategia == EstrategiaEmbedding.ULTIMOS_K:
            emb = self._embedding_last_k(hs, mask)

        emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype(np.float16)

    def guardar(self, embedding: np.ndarray, path: str):
        np.save(path, embedding)

    def cargar(self, path: str) -> np.ndarray:
        return np.load(path)
