# the SepvqvaeR is follow Bailando
# https://github.com/lisiyao21/Bailando
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.distributed as dist
from enum import Enum
import math
import torch.nn.functional as F


class VQVAER(nn.Module):
    def __init__(self, hps, input_dim=72):
        super().__init__()
        self.hps = hps

        input_shape = (hps.sample_length, input_dim)
        levels = hps.levels
        downs_t = hps.downs_t
        strides_t = hps.strides_t
        emb_width = hps.emb_width
        l_bins = hps.l_bins
        mu = hps.l_mu
        commit = hps.commit
        # spectral = hps.spectral
        # multispectral = hps.multispectral
        multipliers = hps.hvqvae_multipliers
        use_bottleneck = hps.use_bottleneck
        if use_bottleneck:
            print('We use bottleneck!')
        else:
            print('We do not use bottleneck!')
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv, \
                            dilation_growth_rate=hps.dilation_growth_rate, \
                            dilation_cycle=hps.dilation_cycle, \
                            reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]
        self.levels = levels

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        encoder = lambda level: Encoder(x_channels, emb_width, level + 1,
                                        downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        decoder = lambda level: Decoder(x_channels, emb_width, level + 1,
                                        downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        decoder_root = lambda level: Decoder(hps.joint_channel, emb_width, level + 1,
                                             downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoders_root = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))
            self.decoders_root.append(decoder_root(level))

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.reg = hps.reg if hasattr(hps, 'reg') else 0
        self.acc = hps.acc if hasattr(hps, 'acc') else 0
        self.vel = hps.vel if hasattr(hps, 'vel') else 0
        if self.reg is 0:
            print('No motion regularization!')
        # self.spectral = spectral
        # self.multispectral = multispectral

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        #xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(zs)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, decoder_root, x_quantised = self.decoders[start_level], self.decoders_root[start_level], xs_quantised[
                                                                                                          0:1]

        x_out = decoder(x_quantised, all_levels=False)
        x_vel_out = decoder_root(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        x_vel_out = self.postprocess(x_vel_out)

        _, _, cc = x_vel_out.size()
        x_out[:, :, :cc] = x_vel_out.clone()
        return x_out,zs

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out,zs = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0),zs

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        #zs = self.bottleneck.encode(xs)
        zs = xs
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x[:, :, :self.hps.joint_channel] = 0
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x):
        self.bottleneck.eval()
        with t.no_grad():

            metrics = {}

            N = x.shape[0]

            x_zero = x.clone()
            x_zero[:, :, :self.hps.joint_channel] = 0

            # Encode/Decode
            x_in = self.preprocess(x_zero)
            xs = []
            for level in range(self.levels):
                encoder = self.encoders[level].eval()
                x_out = encoder(x_in)
                xs.append(x_out[-1])

            zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
            x_outs = []
            x_outs_vel = []

        for level in range(self.levels):
            decoder = self.decoders[level].eval()
            decoder_root = self.decoders_root[level]
            x_out = decoder(xs_quantised[level:level + 1], all_levels=False)
            x_vel_out = decoder_root(xs_quantised[level:level + 1], all_levels=False)

            # x_out[:, :, :cc] = x_vel_out
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)
            x_outs_vel.append(x_vel_out)

        # Loss
        # def _spectral_loss(x_target, x_out, self.hps):
        #     if hps.use_nonrelative_specloss:
        #         sl = spectral_loss(x_target, x_out, self.hps) / hps.bandwidth['spec']
        #     else:
        #         sl = spectral_convergence(x_target, x_out, self.hps)
        #     sl = t.mean(sl)
        #     return sl

        # def _multispectral_loss(x_target, x_out, self.hps):
        #     sl = multispectral_loss(x_target, x_out, self.hps) / hps.bandwidth['spec']
        #     sl = t.mean(sl)
        #     return sl

        recons_loss = t.zeros(()).to(x.device)
        regularization = t.zeros(()).to(x.device)
        velocity_loss = t.zeros(()).to(x.device)
        acceleration_loss = t.zeros(()).to(x.device)
        # spec_loss = t.zeros(()).to(x.device)
        # multispec_loss = t.zeros(()).to(x.device)
        x_target = x.float()[:, :, :self.hps.joint_channel]
        # x_target = audio_postprocess(x.float(), self.hps)[:, :, :self.hps.joint_channel]

        for level in reversed(range(self.levels)):
            x_out_vel = self.postprocess(x_outs_vel[level])
            x_out = self.postprocess(x_outs[level])
            # x_out_vel = audio_postprocess(x_out_vel, self.hps)
            # x_out = audio_postprocess(x_out, self.hps)
            _, _, cc = x_out_vel.size()
            x_out[:, :, :cc] = x_out_vel

            # this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps)
            this_recons_loss = _loss_fn(x_target, x_out_vel)
            # this_spec_loss = _spectral_loss(x_target, x_out, hps)
            # this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            # metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            # metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            # spec_loss += this_spec_loss
            # multispec_loss += this_multispec_loss
            # regularization += t.mean((x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1])**2)

            acceleration_loss += _loss_fn(x_out_vel[:, 1:] - x_out_vel[:, :-1], x_target[:, 1:] - x_target[:, :-1])
            # acceleration_loss +=  _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1], x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
        # if not hasattr(self.)
        # commit_loss = sum(commit_losses)
        # loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss
        loss = recons_loss + self.acc * acceleration_loss
        # +  commit_loss * self.commit + self.reg * regularization + self.vel * velocity_loss + self.acc * acceleration_loss

        with t.no_grad():
            # sc = t.mean(spectral_convergence(x_target, x_out, hps))
            # l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn(x_target, x_out_vel)

            # linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            # spectral_loss=spec_loss,
            # multispectral_loss=multispec_loss,
            # spectral_convergence=sc,
            # l2_loss=l2_loss,
            l1_loss=l1_loss,
            # linf_loss=linf_loss,
            # commit_loss=commit_loss,
            # regularization=regularization,
            velocity_loss=l1_loss,
            acceleration_loss=acceleration_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics


def average_metrics(_metrics):
    metrics = {}
    for _metric in _metrics:
        for key, val in _metric.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)
    return {key: sum(vals)//len(vals) for key, vals in metrics.items()}

class BottleneckBlock(nn.Module):
    def __init__(self, k_bins, emb_width, mu):
        super().__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu
        self.reset_k()
        self.threshold = 1.0

    def reset_k(self):
        self.init = False
        self.k_sum = None
        self.k_elem = None
        self.device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
        self.register_buffer('k', t.zeros(self.k_bins, self.emb_width))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + t.randn_like(x) * std
        return x

    def init_k(self, x):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        # init k_w using random vectors from x
        y = self._tile(x)
        _k_rand = y[t.randperm(y.shape[0])][:k_bins]
        # dist.broadcast(_k_rand, 0)
        self.k = _k_rand
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k
        self.k_elem = t.ones(k_bins, device=self.k.device)

    def restore_k(self, num_tokens=None, threshold=1.0):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k.clone()
        self.k_elem = t.ones(k_bins, device=self.k.device)
        if num_tokens is not None:
            expected_usage = num_tokens / k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)
        self.threshold = threshold

    def update_k(self, x, x_l):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        with t.no_grad():
            # Calculate new centres
            x_l_onehot = t.zeros(k_bins, x.shape[0], device=x.device)  # k_bins, N * L
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)

            _k_sum = t.matmul(x_l_onehot, x)  # k_bins, w
            _k_elem = x_l_onehot.sum(dim=-1)  # k_bins
            y = self._tile(x)
            _k_rand = y[t.randperm(y.shape[0])][:k_bins]

            # dist.broadcast(_k_rand, 0)
            # dist.all_reduce(_k_sum)
            # dist.all_reduce(_k_elem)

            # Update centres
            old_k = self.k
            self.k_sum = mu * self.k_sum + (1. - mu) * _k_sum  # w, k_bins
            self.k_elem = mu * self.k_elem + (1. - mu) * _k_elem  # k_bins
            usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float()
            self.k = usage * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1)) \
                     + (1 - usage) * _k_rand
            _k_prob = _k_elem / t.sum(_k_elem)  # x_l_onehot.mean(dim=-1)  # prob of each bin
            entropy = -t.sum(_k_prob * t.log(_k_prob + 1e-8))  # entropy ie how diverse
            used_curr = (_k_elem >= self.threshold).sum()
            usage = t.sum(usage)
            dk = t.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy,
                    used_curr=used_curr,
                    usage=usage,
                    dk=dk)

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, k_bins)

        if x.shape[-1] == self.emb_width:
            prenorm = t.norm(x - t.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[...,:self.emb_width], x[...,self.emb_width:]
            prenorm = (t.norm(x1 - t.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (t.norm(x2 - t.mean(x2)) / np.sqrt(np.prod(x2.shape)))

            # Normalise
            x = x1 + x2
        else:
            assert False, f"Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}"
        return x, prenorm

    def postprocess(self, x_l, x_d, x_shape):
        # [NT, C] -> NTC -> NCT
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        x_l = x_l.view(N, T)
        return x_l, x_d

    def quantise(self, x):
        # Calculate latent code x_l
        k_w = self.k.t()
        distance = t.sum(x ** 2, dim=-1, keepdim=True) - 2 * t.matmul(x, k_w) + t.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        min_distance, x_l = t.min(distance, dim=-1)

        fit = t.mean(min_distance)
        return x_l, fit

    def dequantise(self, x_l):
        x = F.embedding(x_l, self.k)
        return x

    def encode(self, x):
        N, width, T = x.shape

        # Preprocess.
        x, prenorm = self.preprocess(x)

        # Quantise
        x_l, fit = self.quantise(x)

        # Postprocess.
        x_l = x_l.view(N, T)
        return x_l

    def decode(self, x_l):
        N, T = x_l.shape
        width = self.emb_width

        # Dequantise
        x_d = self.dequantise(x_l)

        # Postprocess
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        return x_d

    def forward(self, x, update_k=True):
        N, width, T = x.shape

        # Preprocess
        x, prenorm = self.preprocess(x)

        # Init k if not inited
        if update_k and not self.init:
            self.init_k(x)

        # Quantise and dequantise through bottleneck
        x_l, fit = self.quantise(x)
        x_d = self.dequantise(x_l)

        # Update embeddings
        if update_k:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}

        # Loss
        commit_loss = t.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_l, x_d = self.postprocess(x_l, x_d, (N,T))
        return x_l, x_d, commit_loss, dict(fit=fit,
                                           pn=prenorm,
                                           **update_metrics)


class Bottleneck(nn.Module):
    def __init__(self, l_bins, emb_width, mu, levels):
        super().__init__()
        self.levels = levels
        level_block = lambda level: BottleneckBlock(l_bins, emb_width, mu)
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(level_block(level))

    def encode(self, xs):
        zs = [level_block.encode(x) for (level_block, x) in zip(self.level_blocks, xs)]
        return zs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        xs_quantised = [level_block.decode(z) for (level_block, z) in zip(self.level_blocks[start_level:end_level], zs)]
        return xs_quantised

    def forward(self, xs):
        zs, xs_quantised, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            z, x_quantised, commit_loss, metric = level_block(x, update_k=self.training)
            zs.append(z)
            if not self.training:
                # Be extra paranoid and make sure the encoder weights can't
                # change from straight-through estimator
                x_quantised = x_quantised.detach()
            xs_quantised.append(x_quantised)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return zs, xs_quantised, commit_losses, metrics

class NoBottleneckBlock(nn.Module):
    def restore_k(self):
        pass

class NoBottleneck(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.level_blocks = nn.ModuleList()
        self.levels = levels
        for level in range(levels):
            self.level_blocks.append(NoBottleneckBlock())

    def encode(self, xs):
        return xs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        return zs

    def forward(self, xs):
        zero = t.zeros(()).cuda()
        commit_losses = [zero for _ in range(self.levels)]
        metrics = [dict(entropy=zero, usage=zero, used_curr=zero, pn=zero, dk=zero) for _ in range(self.levels)]
        return xs, xs, commit_losses, metrics
class ReduceOp(Enum):
    SUM = 0,
    PRODUCT = 1,
    MIN = 2,
    MAX = 3

    def ToDistOp(self):
        return {
            self.SUM: dist.ReduceOp.SUM,
            self.PRODUCT: dist.ReduceOp.PRODUCT,
            self.MIN: dist.ReduceOp.MIN,
            self.MAX: dist.ReduceOp.MAX
        }[self]

def is_available():
    return dist.is_available()

def get_rank():
    if is_available():
        return _get_rank()
    else:
        return 0

def get_world_size():
    if is_available():
        return _get_world_size()
    else:
        return 1

def barrier():
    if is_available():
        return _barrier()
    #else: do nothing

def all_gather(tensor_list, tensor):
    if is_available():
        return _all_gather(tensor_list, tensor)
    else:
        tensor_list[0] = tensor

def all_reduce(tensor, op=ReduceOp.SUM):
    if is_available():
        return _all_reduce(tensor, op)
    #else: do nothing

def reduce(tensor, dst, op=ReduceOp.SUM):
    if is_available():
        return _reduce(tensor, dst, op)
    #else: do nothing

def broadcast(tensor, src):
    if is_available():
        return _broadcast(tensor, src)
    #else: do nothing

def init_process_group(backend, init_method):
    if is_available():
        return _init_process_group(backend, init_method)
    #else: do nothing

def _get_rank():
    return dist.get_rank()

def _barrier():
    return dist.barrier()

def _get_world_size():
    return dist.get_world_size()

def _all_gather(tensor_list, tensor):
    return dist.all_gather(tensor_list, tensor)

def _all_reduce(tensor, op):
    return dist.all_reduce(tensor, op.ToDistOp())

def _reduce(tensor, dst, op):
    return dist.reduce(tensor, dst, op.ToDistOp())

def _broadcast(tensor, src):
    return dist.broadcast(tensor, src)

def _init_process_group(backend, init_method):
    return dist.init_process_group(backend, init_method)
def checkpoint(func, inputs, params, flag):
    if flag:
        args = inputs + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(t.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with t.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        for i in range(len(ctx.input_tensors)):
            temp = ctx.input_tensors[i]
            ctx.input_tensors[i] = temp.detach()
            ctx.input_tensors[i].requires_grad = temp.requires_grad
        with t.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        input_grads = t.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True)
        del ctx.input_tensors
        del output_tensors
        return (None, None) + input_grads

class ResConvBlock(nn.Module):
    def __init__(self, n_in, n_state):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_in, n_state, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_state, n_in, 1, 1, 0),
        )

    def forward(self, x):
        return x + self.model(x)

class Resnet(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0):
        super().__init__()
        self.model = nn.Sequential(*[ResConvBlock(n_in, int(m_conv * n_in)) for _ in range(n_depth)])

    def forward(self, x):
        return self.model(x)

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
            # nn.Conv1d(n_in, n_state, 3, 1, padding, dilation, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, 1, 1, 0,),
            # nn.Conv1d(n_state, n_in, 1, 1, 0, padding_mode='replicate'),
        )
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)

class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_dilation=False, checkpoint_res=False):
        super().__init__()
        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle
        blocks = [ResConv1DBlock(n_in, int(m_conv * n_in),
                                 dilation=dilation_growth_rate ** _get_depth(depth),
                                 zero_out=zero_out,
                                 res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth))
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            if dist.get_rank() == 0:
                print("Checkpointing convs")
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.checkpoint_res == 1:
            for block in self.blocks:
                x = checkpoint(block, (x, ), block.parameters(), True)
            return x
        else:
            return self.model(x)


def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False


def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True

def zero_grad(model):
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None


def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_state(model):
    return sum(s.numel() for s in model.state_dict().values())
class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    # nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t, padding_mode='replicate'),
                    nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t),
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            # block = nn.Conv1d(width, output_emb_width, 3, 1, 1, padding_mode='replicate')
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class DecoderConvBock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_decoder_dilation=False, checkpoint_res=False):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            # block = nn.Conv1d(output_emb_width, width, 3, 1, 1, padding_mode='replicate')
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out, res_scale=res_scale, reverse_dilation=reverse_decoder_dilation, checkpoint_res=checkpoint_res),
                    nn.ConvTranspose1d(width, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t, pad_t)
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(input_emb_width if level == 0 else output_emb_width,
                                                           output_emb_width,
                                                           down_t, stride_t,
                                                           **block_kwargs_copy)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            xs.append(x)

        return xs

class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        level_block = lambda level, down_t, stride_t: DecoderConvBock(output_emb_width,
                                                          output_emb_width,
                                                          down_t, stride_t,
                                                          **block_kwargs)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)
        # self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1, padding_mode='replicate')
    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x
def dont_update(params):
    for param in params:
        param.requires_grad = False


def update(params):
    for param in params:
        param.requires_grad = True


def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]

def _loss_fn(x_target, x_pred):
    return t.mean(t.abs(x_pred - x_target))


class VQVAE(nn.Module):
    def __init__(self, hps, input_dim=72):
        super().__init__()
        self.hps = hps

        input_shape = (hps.sample_length, input_dim)
        levels = hps.levels
        downs_t = hps.downs_t
        strides_t = hps.strides_t
        emb_width = hps.emb_width
        l_bins = hps.l_bins
        mu = hps.l_mu
        commit = hps.commit
        # spectral = hps.spectral
        # multispectral = hps.multispectral
        multipliers = hps.hvqvae_multipliers
        use_bottleneck = hps.use_bottleneck
        if use_bottleneck:
            print('We use bottleneck!')
        else:
            print('We do not use bottleneck!')
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv, \
                            dilation_growth_rate=hps.dilation_growth_rate, \
                            dilation_cycle=hps.dilation_cycle, \
                            reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]
        self.levels = levels

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        encoder = lambda level: Encoder(x_channels, emb_width, level + 1,
                                        downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        decoder = lambda level: Decoder(x_channels, emb_width, level + 1,
                                        downs_t[:level + 1], strides_t[:level + 1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.reg = hps.reg if hasattr(hps, 'reg') else 0
        self.acc = hps.acc if hasattr(hps, 'acc') else 0
        self.vel = hps.vel if hasattr(hps, 'vel') else 0
        if self.reg is 0:
            print('No motion regularization!')
        # self.spectral = spectral
        # self.multispectral = multispectral

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        #xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(zs)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out,zs

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out,zs = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0),zs

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        #zs = self.bottleneck.encode(xs)
        zs = xs
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x):
        metrics = {}

        N = x.shape[0]

        # Encode/Decode
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])

        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level + 1], all_levels=False)
            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        recons_loss = t.zeros(()).to(x.device)
        regularization = t.zeros(()).to(x.device)
        velocity_loss = t.zeros(()).to(x.device)
        acceleration_loss = t.zeros(()).to(x.device)
        x_target = x.float()

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])

            this_recons_loss = _loss_fn(x_target, x_out)
            # this_spec_loss = _spectral_loss(x_target, x_out, hps)
            # this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            # metrics[f'spectral_loss_l{level + 1}'] = this_spec_loss
            # metrics[f'multispectral_loss_l{level + 1}'] = this_multispec_loss
            recons_loss += this_recons_loss
            # spec_loss += this_spec_loss
            # multispec_loss += this_multispec_loss
            regularization += t.mean((x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1]) ** 2)

            velocity_loss += _loss_fn(x_out[:, 1:] - x_out[:, :-1], x_target[:, 1:] - x_target[:, :-1])
            acceleration_loss += _loss_fn(x_out[:, 2:] + x_out[:, :-2] - 2 * x_out[:, 1:-1],
                                          x_target[:, 2:] + x_target[:, :-2] - 2 * x_target[:, 1:-1])
        # if not hasattr(self.)
        commit_loss = sum(commit_losses)
        # loss = recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss
        loss = recons_loss + commit_loss * self.commit + self.reg * regularization + self.vel * velocity_loss + self.acc * acceleration_loss

        with t.no_grad():
            # sc = t.mean(spectral_convergence(x_target, x_out, hps))
            # l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn(x_target, x_out)

            # linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            # spectral_loss=spec_loss,
            # multispectral_loss=multispec_loss,
            # spectral_convergence=sc,
            # l2_loss=l2_loss,
            l1_loss=l1_loss,
            # linf_loss=linf_loss,
            commit_loss=commit_loss,
            regularization=regularization,
            velocity_loss=velocity_loss,
            acceleration_loss=acceleration_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics



smpl_down = [0, 1, 2, 4, 5, 7, 8, 10, 11]
smpl_up = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


class SepVQVAE(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        # self.cut_dim = hps.up_half_dim
        # self.use_rotmat = hps.use_rotmat if (hasattr(hps, 'use_rotmat') and hps.use_rotmat) else False
        self.chanel_num = hps.joint_channel
        self.vqvae_up = VQVAE(hps.up_half, len(smpl_up) * self.chanel_num)
        self.vqvae_down = VQVAE(hps.down_half, len(smpl_down) * self.chanel_num)

        # self.use_rotmat = hps.use_rotmat if (hasattr(hps, 'use_rotmat') and hps.use_rotmat) else False
        # self.chanel_num = 9 if self.use_rotmat else 3
        print("The UP:")
        print(count_parameters(self.vqvae_up))
        print("The DOWN")
        print(count_parameters(self.vqvae_down))

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        """
        zs are list with two elements: z for up and z for down
        """
        if isinstance(zs, tuple):
            zup = zs[0]
            zdown = zs[1]
        else:
            zup = zs
            zdown = zs
        xup = self.vqvae_up.decode(zup)
        xdown = self.vqvae_down.decode(zdown)
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        x = torch.zeros(b, t, (cup + cdown) // self.chanel_num, self.chanel_num).cuda()
        x[:, :, smpl_up] = xup.view(b, t, cup // self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown // self.chanel_num, self.chanel_num)

        return x.view(b, t, -1)

        # z_chunks = [torch.chunk(z, bs_chunks, dim=0) for z in zs]
        # x_outs = []
        # for i in range(bs_chunks):
        #     zs_i = [z_chunk[i] for z_chunk in z_chunks]
        #     x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
        #     x_outs.append(x_out)

        # return torch.cat(x_outs, dim=0)

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        b, t, c = x.size()
        zup = self.vqvae_up.encode(x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_up].view(b, t, -1),
                                   start_level, end_level, bs_chunks)
        zdown = self.vqvae_down.encode(
            x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_down].view(b, t, -1), start_level, end_level,
            bs_chunks)
        return (zup, zdown)

    def sample(self, n_samples):
        # zs = [torch.randint(0, self.l_bins, size=(n_samples, *z_shape), device='cuda') for z_shape in self.z_shapes]
        xup = self.vqvae_up.sample(n_samples)
        xdown = self.vqvae_up.sample(n_samples)
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        x = torch.zeros(b, t, (cup + cdown) // self.chanel_num, self.chanel_num).cuda()
        x[:, :, smpl_up] = xup.view(b, t, cup // self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown // self.chanel_num, self.chanel_num)
        return x

    def forward(self, x):
        b, t, c = x.size()  # 32 240 72

        x = x.view(b, t, c // self.chanel_num, self.chanel_num)  # chanel_num=3  32 240 24 3
        xup = x[:, :, smpl_up, :].view(b, t, -1)  # 32 240 45
        xdown = x[:, :, smpl_down, :].view(b, t, -1)  # 32 240 27

        x_out_up, loss_up, metrics_up = self.vqvae_up(xup)

        x_out_down, loss_down, metrics_down = self.vqvae_down(xdown)

        _, t, cup = x_out_up.size()
        _, _, cdown = x_out_down.size()

        xout = torch.zeros(b, t, (cup + cdown) // self.chanel_num, self.chanel_num).float()
        xout = xout.to(xup.device)
        xout[:, :, smpl_up] = x_out_up.view(b, t, cup // self.chanel_num, self.chanel_num)
        xout[:, :, smpl_down] = x_out_down.view(b, t, cdown // self.chanel_num, self.chanel_num)

        # xout[:, :, smpl_up] = xup.view(b, t, cup//self.chanel_num, self.chanel_num).float()
        # xout[:, :, smpl_down] = xdown.view(b, t, cdown//self.chanel_num, self.chanel_num).float()

        return xout.view(b, t, -1), (loss_up + loss_down) * 0.5, [metrics_up, metrics_down]
class SepVQVAER(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        # self.cut_dim = hps.up_half_dim
        # self.use_rotmat = hps.use_rotmat if (hasattr(hps, 'use_rotmat') and hps.use_rotmat) else False
        self.chanel_num = hps.joint_channel
        self.vqvae_up = VQVAE(hps.up_half, len(smpl_up) * self.chanel_num)
        self.vqvae_down = VQVAER(hps.down_half, len(smpl_down) * self.chanel_num)


    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        """
        zs are list with two elements: z for up and z for down
        """
        if isinstance(zs, tuple):
            zup = zs[0]
            zdown = zs[1]
        else:
            zup = zs
            zdown = zs
        xup,zsup = self.vqvae_up.decode(zup)
        xdown,zsdown = self.vqvae_down.decode(zdown)
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        x = torch.zeros(b, t, (cup + cdown) // self.chanel_num, self.chanel_num).to(xup.device)
        x[:, :, smpl_up] = xup.view(b, t, cup // self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown // self.chanel_num, self.chanel_num)

        return x.view(b, t, -1),zsup[0],zsdown[0]


    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        b, t, c = x.size()
        zup = self.vqvae_up.encode(x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_up].view(b, t, -1),
                                   start_level, end_level, bs_chunks)
        zdown = self.vqvae_down.encode(
            x.view(b, t, c // self.chanel_num, self.chanel_num)[:, :, smpl_down].view(b, t, -1), start_level, end_level,
            bs_chunks)
        return (zup, zdown)

    def sample(self, n_samples):
        xup = self.vqvae_up.sample(n_samples)
        xdown = self.vqvae_up.sample(n_samples)
        b, t, cup = xup.size()
        _, _, cdown = xdown.size()
        x = torch.zeros(b, t, (cup + cdown) // self.chanel_num, self.chanel_num).cuda()
        x[:, :, smpl_up] = xup.view(b, t, cup // self.chanel_num, self.chanel_num)
        x[:, :, smpl_down] = xdown.view(b, t, cdown // self.chanel_num, self.chanel_num)
        return x

    def forward(self, x):
        b, t, c = x.size()
        x = x.view(b, t, c // self.chanel_num, self.chanel_num)
        xup = x[:, :, smpl_up, :].view(b, t, -1)
        xdown = x[:, :, smpl_down, :].view(b, t, -1)

        self.vqvae_up.eval()
        x_out_up, loss_up, metrics_up = self.vqvae_up(xup)
        x_out_down, loss_down, metrics_down = self.vqvae_down(xdown)

        _, _, cup = x_out_up.size()
        _, _, cdown = x_out_down.size()

        xout = torch.zeros(b, t, (cup + cdown) // self.chanel_num, self.chanel_num).cuda().float()
        xout[:, :, smpl_up] = x_out_up.view(b, t, cup // self.chanel_num, self.chanel_num)
        xout[:, :, smpl_down] = x_out_down.view(b, t, cdown // self.chanel_num, self.chanel_num)


        metrics_up['acceleration_loss'] *= 0
        metrics_up['velocity_loss'] *= 0
        return xout.view(b, t, -1), loss_down, [metrics_up, metrics_down]