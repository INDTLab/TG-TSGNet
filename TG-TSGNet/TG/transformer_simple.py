from collections import deque
from collections.abc import Iterable
from functools import partial
from itertools import islice, cycle

import torch
from torch import Tensor
from typing import Optional
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from TG.reversible import ReversibleSequence, SequentialSequence
from TG.attention import Attention, SparseAttention, SparseConvCausalAttention, SparseAxialCausalAttention
from TG.mamba.mamba_ssm.modules.mamba_simple import Mamba
from TG.mamba.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from rotary_embedding_torch import RotaryEmbedding, broadcat


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth=1):
    return val if isinstance(val, Iterable) else (val,) * depth


# classes
# 在指定维度上计算输入张量的最大值，并用这些最大值除以输入张量的每个元素
class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True).detach()
        return x / maxes


class NonCached(nn.Module):
    """
    A wrapper for layers that don't support the inference cache themselves.
    Reconstructs the full sequence before the layer and
    cuts the suffix of the outputs after the layer.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *, cache=None, cache_key=None, **kwargs):
        n = x.shape[-2]
        if exists(cache):
            if cache_key in cache:
                x = torch.cat([cache[cache_key], x], dim=-2)
            cache[cache_key] = x

        out = self.fn(x, **kwargs)

        return out[:, -n:]


class CachedAs(nn.Module):
    """
    A wrapper that defines a key for the inference cache.
    """

    def __init__(self, cache_key, fn):
        super().__init__()
        self.cache_key = cache_key
        self.fn = fn

    def forward(self, x, *, cache=None, **kwargs):
        return self.fn(x, cache=cache, cache_key=self.cache_key, **kwargs)


# https://arxiv.org/abs/2103.17239
class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


# layer norm

class PreNorm(nn.Module):
    def __init__(self, dim, fn, sandwich=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)


# feed forward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, cache=None, cache_key=None):
        return self.net(x)


# token shift classes

class PreShiftToken(nn.Module):
    def __init__(self, fn, image_size, seq_len):
        super().__init__()
        self.fn = fn
        self.image_size = image_size
        self.seq_len = seq_len
        self.img_seq_len = image_size ** 2
        self.text_len = seq_len - self.img_seq_len + 1

    def forward(self, x, cache=None, cache_key=None, **kwargs):
        seq_len, image_size, text_len = self.seq_len, self.image_size, self.text_len

        if exists(cache) and cache_key in cache:
            offset = cache['offset']
            assert offset >= text_len, "cached inference for text is not supported"
            q = cache[cache_key]
            assert isinstance(q, deque) and len(q) == image_size

            x_top, x_left, *x_pass = x[:, -1].chunk(4, dim=-1)

            q.append((x_top, x_left))
            x_top = q.popleft()[0]
            x_left = q[-2][1]
            if (offset - text_len) % image_size == 0:
                x_left = torch.zeros_like(x_left)

            x = torch.cat((x_top, x_left, *x_pass), dim=-1)
            return self.fn(x[:, None], cache=cache, **kwargs)

        n = x.shape[1]
        padding = seq_len - n + 1

        # if sequence is shorter than the text length, no image tokens to shift

        if n < text_len:
            return self.fn(x, **kwargs)

        # get text and image tokens

        x_text, x_img = x[:, :text_len], x[:, text_len:]
        x_img = F.pad(x_img, (0, 0, 0, padding))
        x_img = rearrange(x_img, 'b (h w) d -> b h w d', h=image_size)

        # shift 1 from the left for text tokens

        x_text_shift, x_text_pass = x_text.chunk(2, dim=-1)
        x_text_shift = F.pad(x_text_shift, (0, 0, 1, -1))
        x_text = torch.cat((x_text_shift, x_text_pass), dim=-1)

        # shift from top, left for image tokens

        x_img_shift_top, x_img_shift_left, *x_img_pass = x_img.chunk(4, dim=-1)
        x_img_shift_left = F.pad(x_img_shift_left, (0, 0, 1, -1))
        x_img_shift_top = F.pad(x_img_shift_top, (0, 0, 0, 0, 1, -1))
        x_img = torch.cat((x_img_shift_top, x_img_shift_left, *x_img_pass), dim=-1)

        # merge text and image sequence back together

        x_img = rearrange(x_img, 'b h w d -> b (h w) d')
        x_img = x_img[:, :-padding]
        x = torch.cat((x_text, x_img), dim=1)

        if exists(cache):
            dummy_top, dummy_left, *_ = x[:, -1].chunk(4, dim=-1)
            dummy_top, dummy_left = torch.zeros_like(dummy_top), torch.zeros_like(dummy_left)

            q = deque()
            x_img = x_img[:, -image_size:]
            for _ in range(image_size - x_img.shape[1]):
                q.append((dummy_top, dummy_left))
            for i in range(x_img.shape[1]):
                q.append(x_img[:, i].chunk(4, dim=-1)[:2])
            cache[cache_key] = q

        return self.fn(x, cache=cache, **kwargs)


# main transformer class

class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            seq_len,
            heads=8,
            dim_head=64,
            ff_mult=4,
            attn_dropout=0.,
            ff_dropout=0.,
            rotary_emb=True,
            image_fmap_size=None,
            causal=True,
            stable=False,
    ):
        super().__init__()
        layers = nn.ModuleList([])

        self.seq_len = seq_len  # 序列长度
        self.image_fmap_size = image_fmap_size  # 图像特征图大小

        # 初始化 Transformer 层
        for ind in range(depth):
            # 全注意力机制，结合归一化
            attn = Attention(dim, seq_len, heads=heads, dim_head=dim_head, dropout=attn_dropout, stable=stable)
            ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)

            # 将归一化应用在每层的注意力和前馈模块上
            layers.append(nn.ModuleList([
                PreNorm(dim, attn),  # 注意力归一化
                PreNorm(dim, ff)  # 前馈归一化
            ]))

        self.layers = nn.Sequential(*layers)

        pos_emb = None
        # 如果 rotary_emb 为真，则生成旋转位置嵌入，计算文本和图像的位置频率，并拼接为 pos_emb
        if rotary_emb:
            # 将 dim_head（通常是模型的每个头的维度）除以 3
            # 得到 rot_dim。这是用于旋转位置编码的维度
            rot_dim = dim_head // 3
            # 计算图像序列长度：图像序列的长度等于图像特征图的大小（image_fmap_size）的平方
            img_seq_len = (image_fmap_size ** 2)
            # 文本序列长度 text_len 等于总序列长度 seq_len 减去图像序列长度 img_seq_len，再加上 1。
            # 这个加一是为了包括可能的分隔标记或填充
            text_len = seq_len - img_seq_len + 1

            # 初始化 RotaryEmbedding 对象 text_pos_emb，用于计算文本序列的旋转位置编码
            text_pos_emb = RotaryEmbedding(dim = rot_dim)
            # 创建图像轴向位置编码器：初始化 RotaryEmbedding 对象 img_axial_pos_emb，用于计算图像特征图的旋转位置编码。
            # freqs_for='pixel' 表示该编码器用于像素级的位置编码。
            img_axial_pos_emb = RotaryEmbedding(dim = rot_dim, freqs_for = 'pixel')

            # 对从 0 到 text_len - 1 的整数应用文本位置编码器，得到文本的旋转位置编码 text_freqs
            text_freqs = text_pos_emb(torch.arange(text_len))
            # 为图像分配远离文本的位置信息：为图像序列分配一个大的位置值（8192），确保它们的编码在位置编码中远离文本的编码。
            # 这种方法可以避免文本和图像位置编码的混淆
            img_to_text_freqs = text_pos_emb(torch.full((img_seq_len,), 8192)) # image is given a position far away from text
            # 拼接文本和图像位置编码
            text_freqs = torch.cat((text_freqs, img_to_text_freqs), dim = 0)

            # 计算图像轴向位置编码
            img_freqs_axial = img_axial_pos_emb(torch.linspace(-1, 1, steps = image_fmap_size))
            # 将图像轴向位置编码转换为二维频率
            img_freqs = broadcat((rearrange(img_freqs_axial, 'i d -> i () d'), rearrange(img_freqs_axial, 'j d -> () j d')), dim = -1)
            img_freqs = rearrange(img_freqs, 'h w d -> (h w) d')

            # 为文本分配远离图像的轴向位置编码
            text_axial_freqs = img_axial_pos_emb(torch.full((text_len,), -10.))  # text is given a position of -10 apart from the image axial positions, which is from range [-1, 1]

            # 拼接轴向位置编码
            text_axial_freqs = torch.cat((text_axial_freqs, text_axial_freqs), dim = -1)

            # 将文本轴向位置编码添加到图像编码中
            img_freqs = torch.cat((text_axial_freqs, img_freqs), dim = 0)

            pos_emb = torch.cat((text_freqs, img_freqs), dim = -1)
            pos_emb = rearrange(pos_emb, 'n d -> () n d')

        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x, **kwargs):

        for layer in self.layers:
            attn, ff = layer
            x = attn(x, rotary_pos_emb=self.pos_emb, **kwargs)
            x = ff(x)
        return x


class Mambablock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            seq_len,
            mamba_depth,
            drop_path: float = 0.,
            image_fmap_size=None,
            ff_mult=4,
            ff_dropout=0.
    ):
        super().__init__()

        self.seq_len = seq_len  # 序列长度
        self.image_fmap_size = image_fmap_size  # 图像特征图大小

        # Mamba 层深度
        self.mamba_depth = mamba_depth
        mamba_drop_path = [x.item() for x in torch.linspace(0, drop_path, self.mamba_depth)]

        # 定义 Mamba 层，每个 Mamba 层后面添加归一化
        self.mamba_blocks = nn.ModuleList([
            nn.ModuleList([
                create_block(
                    dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    drop_path=mamba_drop_path[i],
                ),
                nn.LayerNorm(dim)  # 添加归一化层
            ])
            for i in range(self.mamba_depth)
        ])

        # 最终的前馈层
        self.final_ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)

    def forward(self, x, **kwargs):
        residual = None

        # 通过每个 Mamba 层和归一化层
        for i in range(self.mamba_depth):
            x, residual = auto_grad_checkpoint(self.mamba_blocks[i][0], x, residual)
            x = self.mamba_blocks[i][1](x)  # 在 Mamba 层后应用归一化

        # 通过最后的前馈层
        x = self.final_ff(x)

        return x

#####这一块是mamba粘过来的
class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        # 与常规的前归一化 Transformer 块不同，这里的结构为：Add -> LayerNorm -> Mixer，并返回 hidden_states（混合器的输出）和 residual

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        # 存储 residual_in_fp32 参数值
        self.fused_add_norm = fused_add_norm
        # 存储 fused_add_norm 参数值
        self.mixer = mixer_cls(dim)
        # 一个由 mixer_cls 创建的混合器实例，维度大小为 dim
        self.norm = norm_cls(dim)
        # 一个由 norm_cls 创建的归一化层实例，维度大小为 dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 如果 fused_add_norm 为 True，确保 RMSNorm 被正确导入，并且 self.norm 必须是 nn.LayerNorm 或 RMSNorm 类型
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        # 如果 fused_add_norm 为 False
        if not self.fused_add_norm:
            # 如果 residual 为 None，将 hidden_states 赋值给 residual
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            # 如果 residual_in_fp32 为 True，将 residual 转换为32位浮点数
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        # 如果 fused_add_norm 为 True
        # 那就加归一化函数呗
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            # 判断要不要整残差连接的
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        # return hidden_states
        return hidden_states, residual

    # 用于分配推理缓存，调用混合器的 allocate_inference_cache 方法，并传递相关参数
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# 创建一个啥块
def create_block(
        d_model,  # 模型的维度大小
        ssm_cfg=None,  # SSM（State Space Model）配置
        norm_epsilon=1e-5,  # 用于归一化的 epsilon 值
        drop_path=0.,
        rms_norm=False,  # 是否使用 RMS 归一化
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,  # 层的索引
        device=None,
        dtype=None,
        # bimamba_type="v2",  # Mamba 类的类型
        # bimamba_type="none"
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    # 使用 partial 函数创建一个部分实例化的 Mamba 类，传入 layer_idx、bimamba_type、ssm_cfg 和 factory_kwargs

    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    # 使用 partial 函数创建一个部分实例化的归一化类，如果 rms_norm 为 True，则使用 RMSNorm，否则使用 nn.LayerNorm，并传入 norm_epsilon 和 factory_kwargs

    # 上面那个块 多加了mixer_cls norm_cls
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx

    return block


# 初始化模块的权重
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def auto_grad_checkpoint(module, *args, **kwargs):
    # 检查模块是否启用了 grad_checkpointing 属性，如果启用了，则在前向传播过程中使用梯度检查点；否则，直接调用模块的前向传播
    if getattr(module, 'grad_checkpointing', False):
        if isinstance(module, Iterable):
            gc_step = module[0].grad_checkpointing_step
            return checkpoint_sequential(module, gc_step, *args, **kwargs)
        else:
            return checkpoint(module, *args, **kwargs)
    return module(*args, **kwargs)


# 将一个模块列表按指定步长 step 切分成多个部分，并对每个部分应用梯度检查点
def checkpoint_sequential(functions, step, input, *args, **kwargs):
    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input, *args)
            return input

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    # the last chunk has to be non-volatile
    end = -1
    segment = len(functions) // step
    for start in range(0, step * (segment - 1), step):
        end = start + step - 1
        input = checkpoint(run_function(start, end, functions), input, preserve_rng_state=preserve)
    return run_function(end + 1, len(functions) - 1, functions)(input)