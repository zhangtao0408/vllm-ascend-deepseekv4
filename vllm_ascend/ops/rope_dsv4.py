import math
import os
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch_npu
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.platforms import current_platform


class RopeGlobalState:

    def __init__(self):
        self.static_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.runtime_buffer: Dict[str, Dict[str, Tuple[torch.Tensor,
                                                       torch.Tensor]]] = {}
        self.layer_info: Dict[str, Tuple[str, List[str]]] = {}
        self.registry_summary: Dict[str, set] = {}


_ROPE_STATE = RopeGlobalState()


def _rope_debug_enabled() -> bool:
    return os.getenv("DSV4_ROPE_DEBUG", "0").lower() in ("1", "true", "yes", "on")


def _ensure_runtime_buffer_capacity(config_key: str, group_name: str,
                                    num_tokens: int) -> Tuple[torch.Tensor,
                                                              torch.Tensor]:
    buf_cos, buf_sin = _ROPE_STATE.runtime_buffer[config_key][group_name]
    if num_tokens <= buf_cos.size(0):
        return buf_cos, buf_sin

    new_size = max(num_tokens, buf_cos.size(0) * 2)
    if _rope_debug_enabled():
        logger.warning(
            "DSV4_ROPE_DEBUG expanding runtime rope buffer: config_key=%s "
            "group=%s old_size=%s new_size=%s requested_tokens=%s",
            config_key, group_name, buf_cos.size(0), new_size, num_tokens)
    new_cos = torch.ones(new_size,
                         *buf_cos.shape[1:],
                         dtype=buf_cos.dtype,
                         device=buf_cos.device)
    new_sin = torch.zeros(new_size,
                          *buf_sin.shape[1:],
                          dtype=buf_sin.dtype,
                          device=buf_sin.device)
    new_cos[:buf_cos.size(0)].copy_(buf_cos)
    new_sin[:buf_sin.size(0)].copy_(buf_sin)
    _ROPE_STATE.runtime_buffer[config_key][group_name] = (new_cos, new_sin)
    return new_cos, new_sin


class RopeDataProxy:

    def __init__(self, data_map, is_cos=True):
        self._data = data_map
        self.idx = 0 if is_cos else 1

    def __getitem__(self, index):
        if not isinstance(index, str):
            new_map = {}
            for config_k, groups_map in self._data.items():
                new_map[config_k] = {}
                for group_name, item in groups_map.items():
                    c_val = item[0][index]
                    s_val = item[1][index]
                    new_map[config_k][group_name] = (c_val, s_val)

            return RopeDataProxy(new_map, is_cos=(self.idx == 0))

        else:
            layername = index
            info = _ROPE_STATE.layer_info.get(layername)
            if info is None:
                raise KeyError(f"Layer {layername} not registered.")

            config_key, required_groups = info

            config_data = self._data.get(config_key, {})

            layer_result = {}
            for grp in required_groups:
                if grp in config_data:
                    layer_result[grp] = config_data[grp][self.idx]
                else:
                    pass
            if len(layer_result) == 1:
                return list(layer_result.values())[0]

            return layer_result


def get_cos_and_sin_dsa(positions: Union[torch.Tensor, Dict[str,
                                                            torch.Tensor]],
                        use_cache: bool = False):

    if isinstance(positions, torch.Tensor):
        pos_map = {"default": positions}
    else:
        pos_map = positions

    batch_result: dict[Any, Any] = {}

    for config_key, registered_groups in _ROPE_STATE.registry_summary.items():

        if config_key not in _ROPE_STATE.static_cache:
            continue
        static_cos, static_sin = _ROPE_STATE.static_cache[config_key]

        batch_result[config_key] = {}

        for group_name, pos_tensor in pos_map.items():

            if group_name not in registered_groups:
                continue

            curr_cos = static_cos[pos_tensor]
            curr_sin = static_sin[pos_tensor]

            if use_cache:
                group_buffers = _ROPE_STATE.runtime_buffer.get(
                    config_key, {}).get(group_name)

                if group_buffers is None:
                    continue

                num_tokens = pos_tensor.size(0)
                buf_cos, buf_sin = _ensure_runtime_buffer_capacity(
                    config_key, group_name, num_tokens)
                if _rope_debug_enabled():
                    logger.warning(
                        "DSV4_ROPE_DEBUG get_cos_and_sin_dsa cache: "
                        "config_key=%s group=%s positions=%s curr_cos=%s "
                        "buf_cos=%s",
                        config_key, group_name, tuple(pos_tensor.shape),
                        tuple(curr_cos.shape), tuple(buf_cos.shape))

                buf_cos[:num_tokens].copy_(curr_cos)
                buf_sin[:num_tokens].copy_(curr_sin)

                batch_result[config_key][group_name] = (buf_cos[:num_tokens],
                                                        buf_sin[:num_tokens])
            else:
                batch_result[config_key][group_name] = (curr_cos, curr_sin)

    return RopeDataProxy(batch_result,
                         is_cos=True), RopeDataProxy(batch_result,
                                                     is_cos=False)


class ComplexExpRotaryEmbedding(nn.Module):

    def __init__(
        self,
        vllm_config: VllmConfig,
        layername: str,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        scaling_factor: float,
        rope_groups: List[str] = ["default"],
        **extra_kwargs,
    ) -> None:
        super().__init__()
        self.layername = layername
        self.rotary_dim = rotary_dim
        dtype = torch.get_default_dtype()
        beta_fast = extra_kwargs.get("beta_fast", 32)
        beta_slow = extra_kwargs.get("beta_slow", 1)
        config_key = (
            f"rotary_dim{rotary_dim}_max_position_embeddings{max_position_embeddings}_"
            f"base{base}_scaling_factor{scaling_factor}_beta_fast{beta_fast}_beta_slow{beta_slow}"
        )
        _ROPE_STATE.layer_info[layername] = (config_key, rope_groups)

        if config_key not in _ROPE_STATE.registry_summary:
            _ROPE_STATE.registry_summary[config_key] = set()
        for grp in rope_groups:
            _ROPE_STATE.registry_summary[config_key].add(grp)

        if config_key not in _ROPE_STATE.static_cache:
            complex_cis = self.precompute_freqs_cis(rotary_dim,
                                                    max_position_embeddings,
                                                    max_position_embeddings,
                                                    base, scaling_factor,
                                                    beta_fast, beta_slow)
            cos = complex_cis.real.repeat_interleave(2, dim=-1).to(dtype)
            sin = complex_cis.imag.repeat_interleave(2, dim=-1).to(dtype)

            cos = cos.to(current_platform.device_type)
            sin = sin.to(current_platform.device_type)

            _ROPE_STATE.static_cache[config_key] = (
                cos.unsqueeze(1).unsqueeze(1), sin.unsqueeze(1).unsqueeze(1))

        if config_key not in _ROPE_STATE.runtime_buffer:
            _ROPE_STATE.runtime_buffer[config_key] = {}

        target_device = current_platform.device_type
        max_batch_size = vllm_config.scheduler_config.max_num_batched_tokens
        for grp in rope_groups:
            if grp not in _ROPE_STATE.runtime_buffer[config_key]:
                buf_cos = torch.ones(max_batch_size,
                                     1,
                                     1,
                                     rotary_dim,
                                     dtype=dtype,
                                     device=target_device)
                buf_sin = torch.zeros(max_batch_size,
                                      1,
                                      1,
                                      rotary_dim,
                                      dtype=dtype,
                                      device=target_device)
                _ROPE_STATE.runtime_buffer[config_key][grp] = (buf_cos,
                                                               buf_sin)

    @staticmethod
    def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor,
                             beta_fast, beta_slow):

        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return (dim * math.log(max_seq_len /
                                   (num_rotations * 2 * math.pi)) /
                    (2 * math.log(base)))

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(
                find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(
                find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min, max, dim):
            if min == max:
                max += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32) -
                           min) / (max - min)
            return torch.clamp(linear_func, 0, 1)

        freqs = 1.0 / (base
                       **(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if original_seq_len > 0:
            low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                              original_seq_len)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        t = torch.arange(seqlen)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        ori_shape = x.shape
        y = x

        if x.dim() == 2:
            x = x.unsqueeze(-2)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")

        y.copy_(x.view(ori_shape))
        return y

    def extra_repr(self) -> str:
        return f"layername={self.layername}, rotary_dim={self.rotary_dim}"
