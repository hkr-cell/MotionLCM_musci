from typing import Optional, Union

import torch
import torch.nn as nn

from mld.models.operator.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator.attention import (SkipTransformerEncoder,
                                           TransformerEncoderLayer)
from mld.models.operator.utils import get_clones, get_activation_fn, zero_module
from mld.models.operator.position_encoding import build_position_encoding

class MldDenoiser_music(nn.Module):

    def __init__(self,
                 latent_dim: list = [1, 256],
                 hidden_dim: Optional[int] = None,
                 text_dim: int = 768,
                 time_dim: int = 768,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 norm_eps: float = 1e-5,
                 activation: str = "gelu",
                 norm_post: bool = True,
                 activation_post: Optional[str] = None,
                 flip_sin_to_cos: bool = True,
                 freq_shift: float = 0,
                 time_act_fn: str = 'silu',
                 time_post_act_fn: Optional[str] = None,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 force_pre_post_proj: bool = False,
                 text_act_fn: str = 'relu',
                 time_cond_proj_dim: Optional[int] = None,
                 is_controlnet: bool = False) -> None:
        super(MldDenoiser_music, self).__init__()

        self.latent_dim = latent_dim[-1] if hidden_dim is None else hidden_dim
        add_pre_post_proj = force_pre_post_proj or (hidden_dim is not None and hidden_dim != latent_dim[-1])
        self.latent_pre = nn.Linear(latent_dim[-1], self.latent_dim) if add_pre_post_proj else nn.Identity()
        self.latent_post = nn.Linear(self.latent_dim, latent_dim[-1]) if add_pre_post_proj else nn.Identity()

        self.arch = arch
        self.time_cond_proj_dim = time_cond_proj_dim

        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(time_dim, self.latent_dim, time_act_fn,
                                                post_act_fn=time_post_act_fn, cond_proj_dim=time_cond_proj_dim)
        self.emb_proj = nn.Sequential(get_activation_fn(text_act_fn), nn.Linear(text_dim, self.latent_dim))

        self.query_pos = build_position_encoding(self.latent_dim, position_embedding=position_embedding)
        if self.arch == "trans_enc":
            encoder_layer = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
                norm_eps
            )
            encoder_norm = nn.LayerNorm(self.latent_dim, eps=norm_eps) if norm_post and not is_controlnet else None
            self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm,
                                                  activation_post, return_intermediate=is_controlnet)
        else:
            raise ValueError(f"Not supported architecture: {self.arch}!")

        # TODO
        self.is_controlnet = is_controlnet
        if self.is_controlnet:
            self.controlnet_cond_embedding = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim),
                zero_module(nn.Linear(self.latent_dim, self.latent_dim))
            )

            self.controlnet_down_mid_blocks = nn.ModuleList([
                zero_module(nn.Linear(self.latent_dim, self.latent_dim)) for _ in range(num_layers)])

    def forward(self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                timestep_cond: Optional[torch.Tensor] = None,
                controlnet_cond: Optional[torch.Tensor] = None,
                controlnet_residuals: Optional[list[torch.Tensor]] = None
                ) -> Union[torch.Tensor, list[torch.Tensor]]:
        # sample bs 30 512      timestep:[bs]       encoder_hidden_states: bs 30 768        timestep_cond；None


        # 0. dimension matching (pre)




        sample = sample.permute(1, 0, 2)
        sample = sample.to(torch.float32)
        sample = self.latent_pre(sample) # 30 bs 512





        # 1. check if controlnet
        if self.is_controlnet:
            controlnet_cond = controlnet_cond.permute(1, 0, 2)
            sample = sample + self.controlnet_cond_embedding(controlnet_cond)

        # 2. time_embedding

        timesteps = timestep.expand(sample.shape[1]).clone() #将timestep扩展为[bs]，这里没用

        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype) # bs 768

        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb, timestep_cond).unsqueeze(0) # 1 bs 512



        # 3. condition + time embedding
        # text_emb [seq_len, batch_size, text_dim] <= [batch_size, seq_len, text_dim]
        encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2) # 30 bs 768
        # text embedding projection

        text_emb_latent = self.emb_proj(encoder_hidden_states) # 30 bs 512


        emb_latent = torch.cat((time_emb, text_emb_latent), 0) # 31 bs 512


        # 4. transformer
        if self.arch == "trans_enc":
            xseq = torch.cat((sample, emb_latent), axis=0) # 61 bs 512
            xseq = self.query_pos(xseq) # 61 bs 512



            tokens = self.encoder(xseq, controlnet_residuals=controlnet_residuals) # 61 bs 512



            if self.is_controlnet:
                control_res_samples = []
                for res, block in zip(tokens, self.controlnet_down_mid_blocks):
                    r = block(res)
                    control_res_samples.append(r)
                return control_res_samples

            sample = tokens[:sample.shape[0]] # 30 bs 512

        else:
            raise TypeError(f"{self.arch} is not supported")

        # 5. dimension matching (post)
        sample = self.latent_post(sample)
        sample = sample.permute(1, 0, 2) # bs 30 512


        return sample
