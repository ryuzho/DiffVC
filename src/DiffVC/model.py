# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fairseq

from encoder import SpeakerEncoder, ContentFeatureExtractor

from math import sqrt
    


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d
Conv2d = nn.Conv2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer



@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class ConditionVectorUpsampler(nn.Module):
  def __init__(self):
    super().__init__()
    self.ct1 = ConvTranspose2d(1,1,(4,5),stride=5, dilation = (3,1))
    self.c1 = Conv2d(1,1,2,stride = (2,4))
    self.c2 = Conv2d(1,1,2,stride = 2)

  def forward(self, x):
    x = torch.unsqueeze(x, 1) # [8, 1, 99, 256]
    x = self.ct1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.c1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.c2(x)
    x = F.leaky_relu(x, 0.4)
    x = self.c2(x)
    x = F.leaky_relu(x, 0.4)
    x = torch.transpose(x,2,3)
    x = torch.flatten(x, 2, 3)

    return x



class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, uncond=False):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    if not uncond: # conditional model
      self.conditioner_projection = Conv1d(1, 2 * residual_channels, 1)
    else: # unconditional model
      self.conditioner_projection = None

    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, conditioner=None): 
    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    if self.conditioner_projection is None: # using a unconditional model
      y = self.dilated_conv(y)
    else:
      conditioner = self.conditioner_projection(conditioner)
      # dilated_conv(y).shape : [8, 128, 4960]
      y = self.dilated_conv(y) + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)
    
    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class DiffVC(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(1, params.residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))

    self.speaker_encoder = SpeakerEncoder(params.c_in,params.c_h,params.c_out,params.kernel_size,params.bank_size,params.bank_scale,params.c_bank,params.n_conv_blocks,params.n_dense_blocks,params.subsample,params.act,params.dropout_rate)
    self.content_feature_ext = ContentFeatureExtractor(torch.device('cuda'))
    self.condition_vector_upsampler = ConditionVectorUpsampler()
  
    
    self.residual_layers = nn.ModuleList([
        ResidualBlock(params.n_mels, params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
        for i in range(params.residual_layers)
    ])
    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)

  def forward(self, noisy_spectrogram, diffusion_step, spectrogram=None, audio=None):

    # noisy_spectrogram.shape [8, 80, 62]
  
    # audio.shape : [8, 15872]
    x = noisy_spectrogram.flatten(1).unsqueeze(1)
    x = self.input_projection(x)
    x = F.relu(x)
    # x.shape : [8, 64, 80x62]

    # diffusion_step : [ 9, 38, 8, 29, .. 13] 8개 randn
    diffusion_step = self.diffusion_embedding(diffusion_step)
    # diffusion_step.shape : [8, 512]

    speaker_vector = self.speaker_encoder(spectrogram) # speaker_vector.shape : [8, 256]
    speaker_vector = speaker_vector.unsqueeze(1)
    
    content_vector = self.content_feature_ext.get_feature(audio)

    condition_vector = speaker_vector + content_vector
    condition_vector = self.condition_vector_upsampler(condition_vector)
    # condition_vector.shape : [8, 1, 62 * 80]

    skip = None
    for layer in self.residual_layers:
      x, skip_connection = layer(x, diffusion_step, condition_vector)
      skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    
    # print(x.shape) #[8,1,4960]
    
    return x.reshape(x.shape[0],self.params.n_mels,-1)
