"""Encoder BERT for models: SimKGC."""

from abc import ABC
from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
import tqdm
from transformers import AutoConfig, AutoModel

from base.model import BaseModel
from data.dataloader import collate
from data.dataset import Dataset, Example
from models.losses.infonce_loss import compute_infonce_logits
from utils.device import move_to_cuda


def build_model(args) -> nn.Module:
    """Factory function to build the BERT-based model."""

    return CustomBertModel(args)


class CustomBertModel(BaseModel, ABC):
    """BERT-based model architecture for SimKGC, supporting various pooling strategies and negative sampling techniques."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.bert_encoder)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer('pre_batch_vectors', nn.functional.normalize(random_vector, dim=1), persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.bert_encoder)
        self.tail_bert = deepcopy(self.hr_bert)

    def _encode(self, encoder, token_ids, mask, token_type_ids) -> torch.Tensor:
        """Encode input sequences using the specified BERT encoder and pooling strategy."""

        outputs = encoder(
            input_ids=token_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(
        self,
        hr_token_ids,
        hr_mask,
        hr_token_type_ids,
        tail_token_ids,
        tail_mask,
        tail_token_type_ids,
        head_token_ids,
        head_mask,
        head_token_type_ids,
        only_ent_embedding=False,
        **kwargs,
    ) -> dict:
        """Forward pass to compute entity representations and optionally predict entity embeddings directly."""

        if only_ent_embedding:
            return self.predict_ent_embedding(
                tail_token_ids=tail_token_ids,
                tail_mask=tail_mask,
                tail_token_type_ids=tail_token_type_ids,
            )

        hr_vector = self._encode(
            self.hr_bert,
            token_ids=hr_token_ids,
            mask=hr_mask,
            token_type_ids=hr_token_type_ids,
        )

        tail_vector = self._encode(
            self.tail_bert,
            token_ids=tail_token_ids,
            mask=tail_mask,
            token_type_ids=tail_token_type_ids,
        )

        head_vector = self._encode(
            self.tail_bert,
            token_ids=head_token_ids,
            mask=head_mask,
            token_type_ids=head_token_type_ids,
        )

        return {
            'hr_vector': hr_vector,
            'tail_vector': tail_vector,
            'head_vector': head_vector,
        }

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        """Compute logits for contrastive learning based on the encoded representations and the specified loss configuration."""

        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size, device=hr_vector.device)

        logits = compute_infonce_logits(
            query_vec=hr_vector,
            candidate_vec=tail_vector,
            temp=self.log_inv_t,
            margin=self.add_margin if self.training else 0.0,
        )

        return {
            'logits': logits,
            'labels': labels,
            'inv_t': self.log_inv_t.detach().exp(),
            'hr_vector': hr_vector.detach(),
            'tail_vector': tail_vector.detach(),
        }

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        """Predict entity embeddings directly from tail entity inputs, used for efficient inference."""
        ent_vectors = self._encode(
            self.tail_bert,
            token_ids=tail_token_ids,
            mask=tail_mask,
            token_type_ids=tail_token_type_ids,
        )
        return {'ent_vectors': ent_vectors.detach()}

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example], batch_size: Optional[int] = None, num_workers: int = 1) -> (torch.Tensor, torch.Tensor):
        """Predict head-relation and tail entity vectors for a list of examples, used for evaluation."""

        if batch_size is None:
            batch_size = max(self.args.batch_size, 512)

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=self.args.dataset),
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate,
            shuffle=False,
        )

        hr_tensor_list, tail_tensor_list = [], []
        use_cuda = torch.cuda.is_available()
        for _, batch_dict in enumerate(data_loader):
            if use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
            tail_tensor_list.append(outputs['tail_vector'])

        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

    @torch.no_grad()
    def predict_by_entities(self, entity_exs, batch_size: Optional[int] = None, num_workers: int = 2) -> torch.Tensor:
        """Predict entity embeddings for a list of entity examples, used for efficient inference of all entities."""

        examples = []
        for entity_ex in entity_exs:
            entity_id = getattr(entity_ex, 'entity_id', None)
            if entity_id is None:
                entity_id = getattr(entity_ex, 'tail_id', None)
            if entity_id is None:
                raise AttributeError('Expected entity examples with an entity_id or tail_id attribute')
            examples.append(Example(head_id='', relation='', tail_id=entity_id))
        if batch_size is None:
            batch_size = max(self.args.batch_size, 1024)

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=self.args.dataset),
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=collate,
            shuffle=False,
        )

        ent_tensor_list = []
        use_cuda = torch.cuda.is_available()
        for _, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            if use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)


def _pool_output(pooling: str, cls_output: torch.Tensor, mask: torch.Tensor, last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Pool the output of the BERT encoder according to the specified pooling strategy."""

    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
