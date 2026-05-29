"""DaBR encoder adapted from classic DaBR implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel


def build_model(args) -> nn.Module:
    return DaBREncoder(args)


class DaBREncoder(BaseModel):
    """DaBR encoder adapted from classic DaBR implementation.

    Exposes quaternion utilities and a `forward(batch_dict)` returning scores
    and embeddings for regularization.
    """

    def __init__(self, args):
        super().__init__()
        self.config = args
        dim = getattr(args, 'dim', getattr(args, 'hidden_size', 100))
        emb_dim = 4 * int(dim)
        n_ent = getattr(args, 'ent_total', None)
        n_rel = getattr(args, 'rel_total', None)
        # Fallback: many training code builds embeddings later; if counts are missing,
        # create placeholders and expect calling code to set them.
        self.ent_embeddings = nn.Embedding(n_ent or 1, emb_dim)
        self.rel_embeddings = nn.Embedding(n_rel or 1, emb_dim)
        self.Dr = nn.Embedding(n_rel or 1, emb_dim)
        self.para = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize model parameters using Xavier uniform initialization."""

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.Dr.weight.data)

    @staticmethod
    def normalization(quaternion, split_dim=1) -> torch.Tensor:
        """Normalize a quaternion tensor."""

        size = quaternion.size(split_dim) // 4
        quaternion = quaternion.reshape(-1, 4, size)
        quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))
        quaternion = quaternion.reshape(-1, 4 * size)
        return quaternion

    @staticmethod
    def make_wise_quaternion(quaternion) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a quaternion tensor into its component parts."""

        if len(quaternion.size()) == 1:
            quaternion = quaternion.unsqueeze(0)
        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        r2 = torch.cat([r, -i, -j, -k], dim=1)
        i2 = torch.cat([i, r, -k, j], dim=1)
        j2 = torch.cat([j, k, r, -i], dim=1)
        k2 = torch.cat([k, -j, i, r], dim=1)
        return r2, i2, j2, k2

    @staticmethod
    def get_quaternion_wise_mul(quaternion) -> torch.Tensor:
        """Compute the element-wise multiplication of a quaternion tensor."""

        size = quaternion.size(1) // 4
        quaternion = quaternion.view(-1, 4, size)
        quaternion = torch.sum(quaternion, 1)
        return quaternion

    @staticmethod
    def vec_vec_wise_multiplication(q, p) -> torch.Tensor:
        """Compute the element-wise multiplication of two quaternion tensors."""

        normalized_p = DaBREncoder.normalization(p)
        q_r, q_i, q_j, q_k = DaBREncoder.make_wise_quaternion(q)

        qp_r = DaBREncoder.get_quaternion_wise_mul(q_r * normalized_p)
        qp_i = DaBREncoder.get_quaternion_wise_mul(q_i * normalized_p)
        qp_j = DaBREncoder.get_quaternion_wise_mul(q_j * normalized_p)
        qp_k = DaBREncoder.get_quaternion_wise_mul(q_k * normalized_p)

        return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)

    @staticmethod
    def get_inv(quaternion) -> torch.Tensor:
        """Compute the inverse of a quaternion tensor."""

        q_r, q_i, q_j, q_k = torch.chunk(quaternion, 4, dim=1)
        quaternion_norm = q_r ** 2 + q_i ** 2 + q_j ** 2 + q_k ** 2
        r_inv = torch.cat([q_r / quaternion_norm, -q_i / quaternion_norm, -q_j / quaternion_norm, -q_k / quaternion_norm], dim=1)
        return r_inv

    @staticmethod
    def _calc(h, r, t, dr, para) -> torch.Tensor:
        """Calculate the DaBR score for a batch of triples."""

        hr = DaBREncoder.vec_vec_wise_multiplication(h, r)
        r_inv = DaBREncoder.get_inv(r)
        tr = DaBREncoder.vec_vec_wise_multiplication(t, r_inv)
        score_s = hr * tr
        hrt = h + dr - t
        s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=1)
        score_d = s_d + x_d + y_d + z_d
        return -torch.sum(score_s, -1) - para * torch.norm(score_d, p=1, dim=-1)

    @staticmethod
    def regularization(quaternion) -> torch.Tensor:
        """Compute the regularization term for a quaternion tensor."""

        size = quaternion.size(1) // 4
        r, i, j, k = torch.split(quaternion, size, dim=1)
        return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)

    def forward(self, batch_dict: dict) -> dict:
        """Compute scores and embeddings for a batch of triples."""

        h = self.ent_embeddings(batch_dict['head_id'])
        r = self.rel_embeddings(batch_dict['relation'])
        t = self.ent_embeddings(batch_dict['tail_id'])
        dr = self.Dr(batch_dict['relation'])
        score = DaBREncoder._calc(h, r, t, dr, self.para)
        return {
            'scores': score,
            'ent_emb': (h, t),
            'rel_emb': (r, dr),
        }
