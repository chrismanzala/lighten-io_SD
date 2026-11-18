"""
UNET.py - U-Net statistical downscaling (PyTorch)
=================================================

Purpose
-------
Downscale daily climate variables over La Reunion from the CNRM-ESM2-1 GCM grid (~1.4 deg)
to 3 km, using the bias-corrected ALADIN data (ALADIN_CORRIGE) and a DEM.

Inputs / output
---------------
  GCM-grid input (2D, small) : each predictor upscaled onto the CNRM-ESM2-1 grid: mean of the 3 km
             land pixels inside each 1.4 deg cell. La Reunion is covered by 2 x 2 GCM cells, so the
             input is a map of 4 variables x 2 rows x 2 cols per day (rows south -> north, cols
             west -> east). Joined at the bottleneck, where the U-Net map is also 2 x 2.
  1D input : sin and cos of the day of the year (day 1 ... 365, or 366 in leap years), so the
             model knows the season; 31 December and 1 January are neighbours. Goes through
             InputBlock1D (dense stack) and is copied to every bottleneck pixel. --no-seasonal
             leaves it out.
  2D input : static DEM (GEBCO land points only, averaged onto the 3 km grid; every sea pixel of
             DEM/fracLand_REUNION.nc (0 = sea, 1 = land) set to 0 m), one channel.
             Goes through InputBlock2D and the skip connections.
  output   : target variable on the 3 km grid, (N, 1, H, W).

Architecture (default: 32 base channels, 4 blocks)
--------------------------------------------------
  DEM -> InputBlock2D (32) -> 4 x ContractiveBlock (MaxPool + 2 x Conv3x3-BN-ReLU)
                                 channels 64 / 128 / 256 / 256, resolution 32 -> 16 -> 8 -> 4 -> 2
  CoarseBottleneckBlock at 2 x 2:
      terrain features                       256 channels
    + GCM grid (4 x 2 x 2) -> 1x1 conv       128 channels   (--coarse-channels)
    + sin/cos -> InputBlock1D 2->32->64->128 128 channels, copied to the 4 pixels (--dense-neurons)
    = 512 channels -> 2 x Conv3x3-BN-ReLU
  -> 4 x ExpansiveBlock (bilinear Upsample x2 -> Conv3x3 -> concat skip -> 2 x Conv3x3)
                                 channels 256 / 128 / 64 / 32, resolution 4 -> 8 -> 16 -> 32
  -> OutputBlock (1x1 Conv) -> crop to the 28 x 32 grid

  Building blocks (bottom -> top):
    DenseLayer, ConvolutionalLayer, ConvolutionalBlock, ContractiveBlock,
    DoubleResolutionLayer, ConcatChannelsLayer, ExpansiveBlock, BottleNeckBlock,
    CoarseBottleneckBlock, InputBlock2D, InputBlock1D, OutputBlock, UNET.
  UNET.forward(x_2D, x_1D, x_coarse). With input_shape_coarse2D the coarse field (and the 1D
  vector, if any) joins at the bottleneck through CoarseBottleneckBlock; with only
  input_shape_1D it is the older flat-vector BottleNeckBlock; with neither, a plain U-Net.
  H and W must be multiples of 2**number_of_blocks: DownscalingModel pads the static
  DEM and crops the output back to the 3 km grid.

Older runs (e.g. run1) used one flat vector (variables x 4 cells) instead of the 2D GCM grid.
build_model() reads normalization_stats.json and rebuilds either layout, and load_data() /
load_data_cached() still return the flat vector, so gcm_predict.py and predictor_analysis.py
keep working with those runs.

Training and evaluation
-----------------------
  historical (1981-2014) and ssp370 (2015-2100) are joined into one daily series.
  Fixed test set, the same for every fold and never seen in training or early stopping:
    2011-2014 (end of historical), 2046-2050 (mid-century), 2096-2100 (end of century).
  10-fold cross-validation on the remaining 106 years, without any year repeated:
    historical (1981-2010, 30 years = 28.3 % of the non-test data) -> 10 groups of 3 years
    projection (2015-2045 and 2051-2095, 76 years = 71.7 %)        -> 10 groups of 7-8 years
      (4 groups in 2015-2045, 6 in 2051-2095: no group straddles a test period)
  The validation set of fold k = historical group k + projection group k (about 10.6 years,
  28 % / 72 %, the same mix as the data). Every non-test year is a validation year exactly
  once. All other non-test days are the fold's training set (about 35,000 days). Every fold
  is scored on the same test set, and the mean of the folds' predictions (ensemble) is
  scored too. Normalization statistics come from the fold's training days only.
  Loss: masked MSE on land pixels.

Data (under --data-dir, default ~/nas/)
----------------------------------------------
  METEO_FRANCE/DONNEES_SIMULATIONS/ALADIN/ALADIN_CORRIGE/{historical,ssp370}/day/{var}/*.nc
  CMIP_CNRM_ESM21/historical/tas_day_*.nc   (only its lat/lon: the GCM grid)
  DEM/gebco_2025_*.nc                       (DEM, variable "elevation"; negative over the sea)
  DEM/fracLand_REUNION.nc                   (land mask, variable "fracLand": 0 sea, 1 land)
  Predictors: tasmin, hursmax, hursmin, rsds (the *Adjust variables, see PREDICTORS);
  default target: tasmaxAdjust. With --target, the target is removed from the predictors automatically
  (e.g. --target tasminAdjust drops tasminAdjust from the default list); --keep-target-predictor
  keeps it anyway (leakage: the upscaled target is then in the input, scores are optimistic).

Usage
-----
  python UNET.py                          # all 10 folds
  python UNET.py --folds 0 5 9            # only some folds (indices printed at start)
  python UNET.py --test-periods 2011-2014 2046-2050 2096-2100 --n-folds 10
  python UNET.py --target tasminAdjust --batch-size 64
  python UNET.py --help                   # every option with its default

Running on another machine (e.g. a GPU server with the NAS mounted through SSHFS)
---------------------------------------------------------------------------------
  Nothing in the code is tied to one machine: paths, device and threads come from flags
  or environment variables (flag > environment variable > default).

    --data-dir      UNET_DATA_DIR    NAS data root (default /mnt/nas/Data)
    --models-dir    UNET_MODELS_DIR  outputs (default models/unet_3km, relative to the current folder)
    --cache-dir     UNET_CACHE_DIR   local cache of the prepared arrays (default: off)
    --device        UNET_DEVICE      auto | cuda | cuda:1 | mps | cpu (default auto: GPU if found)
    --threads       UNET_THREADS     CPU threads for torch (default: torch's choice)
    --aladin-dir, --dem-path, --gcm-grid-file   override single locations inside the data root

  Example:
    sshfs user@nas:/path/to/Data ~/nas_data -o ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
    pip install -r requirements.txt       # torch: pick the build matching the server's CUDA
    export UNET_DATA_DIR=~/nas_data UNET_CACHE_DIR=~/unet_cache UNET_MODELS_DIR=~/unet_runs/run1
    python UNET.py --dry-run              # checks the mount, reads the data once, fills the cache
    nohup python UNET.py --amp --batch-size 64 > train.log 2>&1 &

  Notes:
    - Reading the NetCDF files through SSHFS is the slow part (several GB). With --cache-dir the
      prepared arrays (~200 MB) are stored on the local disk and later runs skip the NAS.
    - Write outputs to a local disk (--models-dir) rather than to the SSHFS mount.
    - On GPU the whole normalized dataset is moved to GPU memory once (~160 MB).
    - --amp uses mixed precision (bfloat16 when the GPU supports it) for faster training.
    - run_config.json in --models-dir records the options, host, torch version and device used.

Known limitations
-----------------
  - Upscaled ALADIN is a stand-in for the GCM: real CNRM-ESM2-1 cells also include ocean.
  - ssp370 files mix ALADIN63 and ALADIN64 runs (see file names).
  - Neighbouring blocks share a boundary: the last days of one block and the first days of
    the next are correlated (weather persists a few days). Small effect with 5-year blocks.
  - Training years surround each test period (e.g. 2010 and 2015 around 2011-2014), so the
    test periods are interpolation in time, not extrapolation beyond the training climate.
  - ContractiveBlock loop in UNET.__init__: block 0 is built then overwritten by the
    `else` branch, so it does not use replicate padding.
  - DoubleResolutionLayer ignores `output_padding`; ExpansiveBlock does not pass
    `dropout_rate` to it.
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class DenseLayer(nn.Sequential): # maybe put activation function as ReLu
    def __init__(self, in_features, out_features, dropout_rate = 0) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features = in_features, out_features = out_features),
            nn.BatchNorm1d(num_features = out_features),
            nn.ReLU(),
            nn.Dropout(p = dropout_rate)     # (N, F) input: Dropout1d would treat it as unbatched (C, L)
            )


class ConvolutionalLayer(nn.Sequential): # maybe put activation function as ReLu
    def __init__(self, in_channels, out_channels, kernel_size = 3, dropout_rate = 0, 
                 padding = 'same', padding_mode = "zeros",
                 **kwargs) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size = kernel_size, padding = padding, padding_mode = padding_mode,
                      in_channels = in_channels, out_channels = out_channels, bias = False, # batchnorm already has its own bias
                      **kwargs), 
            nn.BatchNorm2d(num_features = out_channels),
            nn.ReLU(),
            nn.Dropout2d(p = dropout_rate)
            )

    

class ConvolutionalBlock(nn.Sequential): # Add an option for Res convolutional block
    def __init__(self, in_channels, kernel_size = 3, multiplies_channels_by = 1, dropout_rate = 0, 
                 padding = 'same', padding_mode = 'zeros') -> None:
        super().__init__()
        
        self.out_channels = int(multiplies_channels_by*in_channels)

        self.block = nn.Sequential(
            ConvolutionalLayer(in_channels = in_channels, out_channels = self.out_channels, kernel_size = kernel_size,  dropout_rate = dropout_rate, 
                               padding = padding, padding_mode = padding_mode),                                 
            ConvolutionalLayer(in_channels = self.out_channels, out_channels = self.out_channels, kernel_size = kernel_size, dropout_rate = dropout_rate,
                               padding = padding, padding_mode = padding_mode)
            # add residual TRUE FALSE; requires changes to non-sequential
            )
            

# Specific to UNET

class ContractiveBlock(nn.Sequential):
    def __init__(self, in_channels, kernel_size = 3, multiplies_channels_by = 2, dropout_rate = 0, padding = 'same', padding_mode = 'zeros') -> None:
        super().__init__()

        convolutional_block = ConvolutionalBlock(in_channels = in_channels,
                                                 dropout_rate = dropout_rate,
                                                 kernel_size = kernel_size,
                                                 multiplies_channels_by = multiplies_channels_by,
                                                 padding = padding,
                                                 padding_mode = padding_mode )
        
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            convolutional_block
            )

        self.out_channels = convolutional_block.out_channels
        


class DoubleResolutionLayer_old(nn.Sequential): # better name upsampling lsyer
    def __init__(self, in_channels, halves_channels = True, output_padding = (0,0),
                dropout_rate = 0) -> None:
        super().__init__()  
        self.out_channels = in_channels
        
        if (halves_channels):   # if (skipConnection_channels > 0) always halves in transposition
            self.out_channels = int(self.out_channels/2)

        
        self.conv_transpose = nn.ConvTranspose2d(in_channels = in_channels,    # In the convTranspose it is common to use 2x2 kernels, 3by3 creates artifacts checkerboard effect
                                                 out_channels = self.out_channels, 
                                                 kernel_size = 2, stride = 2, # 
                                                 output_padding = output_padding)


class DoubleResolutionLayer(nn.Module):
    # Created to remove checkerboard effect of transposed convolution, by first upsampling and then convolving with a normal convolution.
    def __init__(self, in_channels, halves_channels = True, output_padding = (0,0), dropout_rate=0.0) -> None:
        super().__init__()

        out_channels = in_channels // 2 if halves_channels else in_channels
        self.out_channels = out_channels

        self.upsample = nn.Upsample(scale_factor = 2, mode = "bilinear") 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      padding=1, padding_mode = "replicate", 
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class ConcatChannelsLayer(nn.Module):
    def __init__(self, x1_channels, x2_channels) -> None:
        # Set skipConnection_channels = 0 for no skip
        super().__init__()

        self.out_channels = x1_channels + x2_channels

    def forward(self, x_1, x_2):
        x = torch.concat(tensors = (x_1, x_2), dim = 1)

        return(x)
   



#Every step in the expansive path consists of an upsampling of the feature map followed by
#  a 2x2 convolution (“up-convolution”) that halves the number of feature channels, 
# a concatenation with the correspondingly cropped feature map from the contracting path, 
# and two 3x3 convolutions, each followed by a ReLU. 
# The cropping is necessary due to the loss of border pixels in every convolution.



class ExpansiveBlock(nn.Module):
    def __init__(self,
                 in_channels, skip_x_out_channels,
                 halves_channels_in_resolution_increase,
                 halves_channels_after_convolutional_block, 
                 kernel_size = 3,
                 dropout_rate = 0, output_padding = (0,0), 
                 ) -> None:
        # set skip_x_out_channels = 0 for no skip connection

        super().__init__()        
        
        self.skip_x_out_channels = skip_x_out_channels
        

        self.double_resolution_layer = DoubleResolutionLayer(in_channels = in_channels,
                                                             halves_channels = halves_channels_in_resolution_increase,
                                                             output_padding = output_padding)


        if skip_x_out_channels > 0:
            self.concat_channels_layer = ConcatChannelsLayer(x1_channels = self.double_resolution_layer.out_channels,
                                                             x2_channels = skip_x_out_channels)
            in_channels = self.concat_channels_layer.out_channels
        else:
            in_channels = self.double_resolution_layer.out_channels


        if halves_channels_after_convolutional_block:
            multiplies_channels_by = 1/2
        else:
            multiplies_channels_by = 1
            
        self.convolutional_block = ConvolutionalBlock(in_channels = in_channels, kernel_size = kernel_size,
                                                      multiplies_channels_by = multiplies_channels_by,
                                                      dropout_rate = dropout_rate)            

        self.out_channels = self.convolutional_block.out_channels


    def forward(self, x, x_from_skip_connection = None):
        x = self.double_resolution_layer(x)
        
        if self.skip_x_out_channels > 0:
            x = self.concat_channels_layer(x, x_from_skip_connection)
        
        x = self.convolutional_block(x)

        return(x)
        

        


class BottleNeckBlock(nn.Sequential):
    # Goes through a 1D Bottleneck if not already 1D and reverts back to original
    # out_channels is given by the InputBlock1D
    # fixme_low : do sth
    # dimensions_2D_input : does NOT include first dimension; so 0 is channels

    def __init__(self, dimensions_2D_input, input1Dblockoutput_features) -> None:
        super().__init__()

        kernel_size = dimensions_2D_input[1:3]
        in_channels = dimensions_2D_input[0]


        self.unflatten1D = nn.Unflatten(dim = 1, unflattened_size = (input1Dblockoutput_features, 1, 1) )

        # Redouane: " Better expand to MxM, last layer of MLP outputs the unflattened res of the 2D "

        if ( (kernel_size[0]) == 1 and (kernel_size[1] == 1) ):
            self.convolution_to_1_1 = nn.Identity()
            self.moveBacktoOriginal2Dshape = nn.Identity()
        else:
            self.convolution_to_1_1 = nn.Sequential(
                nn.Conv2d(kernel_size = kernel_size, padding = 0, 
                          in_channels = in_channels, out_channels = in_channels,
                          bias = False),  # already a bias term in batchnorm
                nn.BatchNorm2d(num_features = in_channels)
                )
            
            self.moveBacktoOriginal2Dshape = nn.ConvTranspose2d(in_channels = in_channels + input1Dblockoutput_features,
                                                                out_channels = in_channels + input1Dblockoutput_features, 
                                                                kernel_size = kernel_size, 
                                                                stride = 1) 

        self.out_channels = in_channels + input1Dblockoutput_features



    def forward(self, x, input1Dblock_output):
        x = self.convolution_to_1_1(x)
        x1d = self.unflatten1D(input1Dblock_output)
       
        xx = torch.concat(tensors = (x1d, x), dim = 1)

        xx = self.moveBacktoOriginal2Dshape(xx)

        return( xx )



class CoarseBottleneckBlock(nn.Module):
    # Bottleneck for a coarse 2D input, e.g. the GCM cells over the domain (variables x rows x cols).
    # After the contractive path the U-Net map is small (2 x 2 for a 32 x 32 input), about the size of the
    # GCM grid over La Reunion (2 x 2 cells), so the coarse field is joined there as extra channels and keeps
    # the position of each cell. If the two grids differ in size, the coarse field is resized (nearest).
    # An optional 1D vector (e.g. sin/cos of the day of the year, after InputBlock1D) is copied to every
    # bottleneck pixel. A ConvolutionalBlock then mixes terrain, GCM and vector channels.
    def __init__(self, in_channels, bottleneck_hw, coarse_shape, coarse_channels = 128, vector_features = 0,
                 dropout_rate = 0) -> None:
        super().__init__()
        self.bottleneck_hw = tuple(int(v) for v in bottleneck_hw)
        self.coarse_proj = ConvolutionalLayer(in_channels = coarse_shape[0], out_channels = coarse_channels,
                                              kernel_size = 1, padding = 0, dropout_rate = dropout_rate)
        mixed = in_channels + coarse_channels + vector_features
        self.mix = ConvolutionalBlock(in_channels = mixed, kernel_size = 3, multiplies_channels_by = 1,
                                      dropout_rate = dropout_rate, padding = 'same', padding_mode = 'replicate')
        self.out_channels = self.mix.out_channels

    def forward(self, x, x_coarse, x_1D = None):
        if x_coarse.shape[-2:] != x.shape[-2:]:
            x_coarse = F.interpolate(x_coarse, size = x.shape[-2:], mode = "nearest")
        parts = [x, self.coarse_proj(x_coarse)]
        if x_1D is not None:
            parts.append(x_1D[:, :, None, None].expand(-1, -1, *x.shape[-2:]))
        return self.mix(torch.cat(parts, dim = 1))



class InputBlock2D(nn.Sequential):
    def __init__(self, input_shape, reshaped_input_size = None, kernel_size = 3, number_of_kernels = 64, dropout_rate = 0,
                 padding = 'same', padding_mode = 'zeros') -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvolutionalLayer(in_channels = input_shape[1], out_channels = number_of_kernels, kernel_size = kernel_size,
                               dropout_rate = dropout_rate, padding = padding, padding_mode = padding_mode),                                  
            ConvolutionalLayer(in_channels = number_of_kernels, out_channels = number_of_kernels, kernel_size = kernel_size,
                               dropout_rate = dropout_rate, padding = padding, padding_mode = padding_mode)
            )  
        
        self.out_channels = number_of_kernels
        


class InputBlock1D(nn.Sequential):
    def __init__(self, input_shape, neurons_dense_list, dropout_rate = 0) -> None:
        super().__init__()
        
        self.block = nn.Sequential()

        in_features = input_shape[1]
        for features in neurons_dense_list:
            self.block.append( DenseLayer(in_features = in_features, out_features = features, 
                                        dropout_rate = dropout_rate) )
            in_features = features

        self.out_features = features

    

class OutputBlock(nn.Module):
    # fixme_low : I'm inheriting from Module since this has a submodule ExpansiveBlock which is NOT sequential.
    #               But can this be put into a sequential? Just forwarding x and the default for ExpansiveBlock forward
    #                second arg is None
    # 

    def __init__(self,  in_channels, out_channels,
                 kernel_size = 3, double_resolution_times = 0, dropout_rate = 0) -> None:
        # out_channels = int or list, listing activations
        
        super().__init__()

        self.output_block = nn.ModuleList()


        if (double_resolution_times > 0):
            for i in range(double_resolution_times):
                if ( i == (double_resolution_times -1) ):
                    dropout_rate = 0

                block = ExpansiveBlock(in_channels = in_channels,
                                       skip_x_out_channels = 0, 
                                       kernel_size = kernel_size,
                                       halves_channels_after_convolutional_block = False,
                                       halves_channels_in_resolution_increase = False,           
                                       dropout_rate = dropout_rate)
                in_channels = block.out_channels
                self.output_block.append( block )


        # Activations at the output (if any):
        if type(out_channels) is list:
            self.output_activations_layer = nn.ModuleList()
            for out_activation in out_channels:
                self.output_activations_layer.append( out_activation() )

            self.out_channels = len(out_channels)
        else:
            self.output_activations_layer = None                
            self.out_channels = out_channels


        self.output_block.append(
                    nn.Conv2d(in_channels = in_channels, out_channels = self.out_channels, 
                              kernel_size = 1, padding = 0)
                                )


    def forward(self, x): 
        
        for layer in self.output_block:
            x = layer(x) 


        if (self.output_activations_layer is None):
            return(x)

        else:
            xx = self.output_activations_layer[0]( x[:,[0],:,:] )
            for i in range(1, x.shape[1]): # output in conv2d is (N, C, H, W)
                xx = torch.cat( (xx, self.output_activations_layer[i]( x[:,[i], ...])), dim = 1)

            return(xx)    



class UNET(nn.Module):

    def __init__(self, 
                 input_shape_2D, output_channels,
                 input_shape_1D = None, 
                 input_shape_coarse2D = None,
                 coarse_channels = 128,
                 input_processing_block_2D = None,
                 input_number_kernels = 64,
                 number_of_blocks = 4,
                 inputNN_neurons_list = [64, 128, 256, 512],
                 kernel_size = 3,
                 double_resolution_times = 2,
                 dropout_rate_1Dinput = 0.,
                 dropout_rate_2D = 0.,
                 debug_sizes = False) -> None:
        # input_size: (N, Variables, Height,, Width)
        # input_shape_1D       : (N, features) vector joined at the bottleneck (BottleNeckBlock, or CoarseBottleneckBlock)
        # input_shape_coarse2D : (variables, rows, cols) coarse field joined at the bottleneck (CoarseBottleneckBlock)
        # input_processing_block_2D : a block that processes the input 2D data (e.g., cropping)
        #    it requires an output_shape attribute, same format as input_shape_2D, which is typically e.g., xx2d.shape
        super().__init__()

        self.debug_sizes = debug_sizes
        self.number_of_blocks = number_of_blocks

        if not (input_processing_block_2D is None): # Allows for a personalized input block that processes 2d input (e.g., cropping) 
            self.input_processing_block_2D = input_processing_block_2D
            self.input_shape_2D = self.input_processing_block_2D.output_shape
        else:
            self.input_processing_block_2D = None
            self.input_shape_2D = input_shape_2D
            

        self.input_shape_1D = input_shape_1D
        self.input_shape_coarse2D = tuple(input_shape_coarse2D) if input_shape_coarse2D is not None else None
        adds_at_bottleneck = (input_shape_1D is not None) or (self.input_shape_coarse2D is not None)
        # Fix input-output dimensions using output_padding in deconvolutions:
        self.out_padding_pattern = torch.zeros(number_of_blocks, 2)        
        dimension_tracker = torch.tensor(self.input_shape_2D[2:4])

        for i in range(number_of_blocks):
            self.out_padding_pattern[i , :] = (dimension_tracker % 2) != 0
            dimension_tracker = (dimension_tracker - (dimension_tracker % 2)) / 2
            
        self.out_padding_pattern = self.out_padding_pattern.type(torch.int32)



        # The model:
        self.input_block2D = InputBlock2D(input_shape = self.input_shape_2D, kernel_size = kernel_size,
                                          number_of_kernels = input_number_kernels,
                                          dropout_rate = dropout_rate_2D,
                                          padding_mode = "replicate")
        if self.input_shape_1D != None:
            self.input_block1D = InputBlock1D(input_shape = self.input_shape_1D, neurons_dense_list = inputNN_neurons_list,
                                          dropout_rate = dropout_rate_1Dinput)
        else:
            self.input_block1D = None

        #   The Paths:
        in_channels = input_number_kernels
        self.contractivePath = nn.ModuleList()

        skip_x_out_channels_list = [ self.input_block2D.out_channels ] 
        
        for i in range(number_of_blocks):
            if i == 0:
                block = ContractiveBlock( in_channels, kernel_size = kernel_size, dropout_rate = dropout_rate_2D, 
                                         multiplies_channels_by = (1 if adds_at_bottleneck else 2),
                                         padding_mode = "replicate")                
            if i == (number_of_blocks-1):
                block = ContractiveBlock( in_channels, kernel_size = kernel_size, dropout_rate = dropout_rate_2D, 
                                         multiplies_channels_by = (1 if adds_at_bottleneck else 2) )
                #last block is not connected with skip, 
                # also in orignal model description last block does NOT double number of features, because the bottleneck adds them
                # here if 1D input is set to None the channels will be multiplied.
            else: 
                block = ContractiveBlock(in_channels, kernel_size = kernel_size, dropout_rate = dropout_rate_2D, multiplies_channels_by = 2)
                skip_x_out_channels_list.append(block.out_channels)

            in_channels = block.out_channels
            self.contractivePath.append( block )


        
        bottleneck_hw = (int(dimension_tracker[0].item()), int(dimension_tracker[1].item()))
        if self.input_shape_coarse2D is not None:
            self.bottleneck_block = CoarseBottleneckBlock(
                in_channels = in_channels, bottleneck_hw = bottleneck_hw, coarse_shape = self.input_shape_coarse2D,
                coarse_channels = coarse_channels,
                vector_features = self.input_block1D.out_features if self.input_block1D is not None else 0,
                dropout_rate = dropout_rate_2D)
            in_channels = self.bottleneck_block.out_channels
        elif self.input_shape_1D != None:
            self.bottleneck_block = BottleNeckBlock(dimensions_2D_input = (in_channels, 
                                                                           int(dimension_tracker[0].item()),
                                                                           int(dimension_tracker[1].item())),
                                                                           input1Dblockoutput_features = self.input_block1D.out_features)
            in_channels = self.bottleneck_block.out_channels 



        self.expansivePath = nn.ModuleList()


        
        for i in range(number_of_blocks): 
            
            skip_x_out_channels = skip_x_out_channels_list.pop()

            expansive_block = ExpansiveBlock(in_channels = in_channels, 
                                             skip_x_out_channels = skip_x_out_channels,
                                             kernel_size = kernel_size,
                                             halves_channels_in_resolution_increase = True,                                             
                                             halves_channels_after_convolutional_block = True,
                                             output_padding = self.out_padding_pattern[number_of_blocks - (i+1), ].tolist(), 
                                             dropout_rate = dropout_rate_2D)

            self.expansivePath.append(expansive_block)

            in_channels = expansive_block.out_channels


        self.output_block = OutputBlock(in_channels = in_channels, out_channels = output_channels,
                                        kernel_size = kernel_size,
                                        double_resolution_times = double_resolution_times, 
                                        dropout_rate = dropout_rate_2D)





    def forward(self, x, x_1D = None, x_coarse = None):
        
        #print("input shape: ", x.shape)
        if not (self.input_processing_block_2D is None):
            x = self.input_processing_block_2D(x)            
            #print("after input_processing_block_2D shape: ", x.shape)

        x = self.input_block2D(x)
        #print("after input_block2D shape: ", x.shape)
        
        if self.input_block1D != None:
            x_1D = self.input_block1D(x_1D)
            #print("after input_block1D shape: ", x_1D.shape)
        


        skip_x_list = []
        
        for i in range(self.number_of_blocks):
            skip_x_list.append(x.clone())
            #print("skip_x_list added x with shape " + str(skip_x_list[-1].shape))
            x = self.contractivePath[i](x)
            #print("after contractivePath block " + str(i) + " shape: ", x.shape)

        if self.input_shape_coarse2D is not None:
            x = self.bottleneck_block(x, x_coarse, x_1D)
        elif self.input_block1D != None:
            x = self.bottleneck_block(x = x, input1Dblock_output = x_1D)
            #print("after bottleneck_block shape: ", x.shape)

        for expasive_block, skip_x in zip(self.expansivePath, skip_x_list[::-1]): 
            x = expasive_block(x = x, x_from_skip_connection = skip_x)
            #print("after expansivePath block shape: ", x.shape)
            
        x = self.output_block(x)
        #print("after output_block shape: ", x.shape)
        return(x)





# =====================================================================
# Training: GCM-grid (1.4 deg) predictors + DEM  ->  3 km target
#   2D input : static DEM on the 3 km grid  (InputBlock2D + skip connections)
#   1D input : predictors upscaled onto the CNRM-ESM2-1 grid (~1.4 deg), one value per variable
#              and per GCM cell overlapping the domain, flattened (InputBlock1D -> bottleneck)
#   output   : target variable on the 3 km grid (ALADIN_CORRIGE)
# Data: historical (1981-2014) + ssp370 (2015-2100), joined in time.
# Evaluation: blocked time cross-validation, one 5-year test block per fold.
# =====================================================================

# Paths and machine settings. Each one can be set by a command-line flag or an environment variable,
# so the same file runs on the NAS host, on a remote GPU machine with the NAS mounted through SSHFS, etc.
#   priority: command-line flag > environment variable > default below
DEFAULT_DATA_DIR = "~/nas/"                                       # --data-dir    | UNET_DATA_DIR
DEFAULT_MODELS_DIR = "models/unet_3km"                                   # --models-dir  | UNET_MODELS_DIR
DEFAULT_CACHE_DIR = ""                                                   # --cache-dir   | UNET_CACHE_DIR ("" = no cache)
DEFAULT_DEVICE = "auto"                                                  # --device      | UNET_DEVICE
# Locations inside the data directory (override the full path with --aladin-dir, --dem-path, --land-frac-path, --gcm-grid-file)
ALADIN_SUBDIR = "METEO_FRANCE/DONNEES_SIMULATIONS/ALADIN/ALADIN_CORRIGE"  # {scenario}/day/{var}/*.nc
DEM_SUBPATH = "DEM/gebco_2025_n-20.82_s-21.45_w55.15_e55.9.nc"
LAND_FRAC_SUBPATH = "DEM/fracLand_REUNION.nc"                            # fracLand: 0 sea, 1 land (3 km grid)
GCM_GRID_SUBPATH = "CMIP_CNRM_ESM21/historical/tas_day_CNRM-ESM2-1_historical_r1i1p1f2_gr_18500101-20141231.nc"

SCENARIOS = ["historical", "ssp370"]
PREDICTORS = ["tasminAdjust", "hursmaxAdjust", "hursminAdjust","rsdsAdjust"]       # present in both historical and ssp370; target excluded
TARGET = "tasmaxAdjust"
TEST_PERIODS = ["2011-2014", "2046-2050", "2096-2100"]   # fixed test set, the same for every fold
N_FOLDS = 10                                              # cross-validation folds; each non-test year validates once
PROJECTION_START = 2015                                   # first year of the scenario part (historical before it)


def load_variable(aladin_dir, scenario, var):
    import xarray as xr

    folder = Path(aladin_dir) / scenario / "day" / var
    files = sorted(glob.glob(str(folder / "*.nc")))
    if not files:
        raise FileNotFoundError(f"No files for '{var}' in {folder} (check --data-dir / --aladin-dir and the SSHFS mount)")

    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            parts.append(ds[var].load())
    da = xr.concat(parts, dim = "time") if len(parts) > 1 else parts[0]
    da = da.assign_coords(time = da["time"].dt.floor("D"))      # daily data: align on dates only
    return da.transpose("time", "lat", "lon")


def load_variable_all(aladin_dir, scenarios, var):
    import xarray as xr

    da = xr.concat([load_variable(aladin_dir, s, var) for s in scenarios], dim = "time")
    _, keep = np.unique(da["time"].values, return_index = True)   # drop duplicated dates at scenario joins, if any
    return da.isel(time = np.sort(keep))


def load_land_frac(path, lat, lon):
    # Land/sea mask on the 3 km grid from fracLand (0 = sea, 1 = land). Must be on the same grid as ALADIN.
    import xarray as xr

    with xr.open_dataset(path) as ds:
        frac = ds["fracLand"].load()
    if frac.shape != (len(lat), len(lon)) or not (np.allclose(frac["lat"].values, lat, atol = 1e-4)
                                                  and np.allclose(frac["lon"].values, lon, atol = 1e-4)):
        frac = frac.interp(lat = lat, lon = lon, method = "nearest")
    return np.nan_to_num(frac.values) > 0.5                     # (H, W) bool, True = land


def load_dem(path, lat, lon, land_frac_path = None):
    # GEBCO (~450 m) -> 3 km grid. GEBCO is negative over the sea (bathymetry): only land elevations are kept.
    #   1. each 3 km pixel = mean of the GEBCO points > 0 m inside it (cell = centre +- half the grid step),
    #      so a coastal pixel is not pulled down by the sea floor; no GEBCO land point inside -> 0 m
    #   2. with land_frac_path (fracLand: 0 sea, 1 land), every sea pixel is set to 0 m
    #   Pixels outside the DEM extent are 0 m.
    import xarray as xr

    with xr.open_dataset(path) as ds:
        dem = ds["elevation"].load()
    g_lat, g_lon, z = dem["lat"].values, dem["lon"].values, dem.values.astype(np.float64)

    def cell_index(g, c):                                        # 3 km cell of each GEBCO coordinate, -1 = outside
        step = np.abs(np.diff(c)).mean()
        k = np.abs(g[:, None] - c[None, :]).argmin(axis = 1)
        return np.where(np.abs(g - c[k]) <= step / 2, k, -1)

    iy, ix = cell_index(g_lat, lat), cell_index(g_lon, lon)
    keep = (iy[:, None] >= 0) & (ix[None, :] >= 0) & (z > 0)    # land GEBCO points inside the 3 km grid
    flat = (np.broadcast_to(iy[:, None], z.shape) * len(lon) + np.broadcast_to(ix[None, :], z.shape))[keep]
    total = np.bincount(flat, weights = z[keep], minlength = len(lat) * len(lon))
    count = np.bincount(flat, minlength = len(lat) * len(lon))
    out = np.where(count > 0, total / np.maximum(count, 1), 0).reshape(len(lat), len(lon)).astype(np.float32)
    if land_frac_path is not None:
        out[~load_land_frac(land_frac_path, lat, lon)] = 0
    return out                                                   # (H, W)


def gcm_cells(lat, lon, gcm_grid_file):
    # Labels each 3 km pixel with the GCM cell that contains it (nearest GCM cell centre = inside its bounds).
    import xarray as xr

    with xr.open_dataset(gcm_grid_file) as ds:
        g_lat, g_lon = ds["lat"].values, ds["lon"].values % 360
    iy = np.abs(lat[:, None] - g_lat[None, :]).argmin(axis = 1)          # (H,)
    ix = np.abs((lon[:, None] % 360) - g_lon[None, :]).argmin(axis = 1)  # (W,)
    cell = iy[:, None] * len(g_lon) + ix[None, :]                        # (H, W) global cell id
    ids, labels = np.unique(cell, return_inverse = True)
    centres = [(float(g_lat[i // len(g_lon)]), float(g_lon[i % len(g_lon)])) for i in ids]
    return labels.reshape(cell.shape), centres


def upscale_to_gcm(values, labels, n_cells):
    # (T, H, W) 3 km field -> (T, n_cells): mean of the valid (land) pixels inside each GCM cell
    flat = values.reshape(len(values), -1)
    out = np.full((len(values), n_cells), np.nan, dtype = np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)               # cell without land -> NaN
        for k in range(n_cells):
            out[:, k] = np.nanmean(flat[:, labels.ravel() == k], axis = 1)
    return out


def gcm_grid(lat, lon, gcm_grid_file):
    # The GCM cells covering the domain as a small regular grid, oriented like the 3 km data
    # (rows follow the data's latitude order, columns its longitude order). Over La Reunion: 2 x 2 cells.
    labels, centres = gcm_cells(lat, lon, gcm_grid_file)
    c_lat = np.array([c[0] for c in centres]); c_lon = np.array([c[1] for c in centres])
    cell_lat, cell_lon = np.unique(c_lat), np.unique(c_lon)
    if len(centres) != len(cell_lat) * len(cell_lon):
        raise ValueError(f"The {len(centres)} GCM cells over the domain do not form a rectangle")
    if lat[0] > lat[-1]:
        cell_lat = cell_lat[::-1]
    if (lon[0] % 360) > (lon[-1] % 360):
        cell_lon = cell_lon[::-1]
    rows = np.array([np.where(cell_lat == v)[0][0] for v in c_lat])
    cols = np.array([np.where(cell_lon == v)[0][0] for v in c_lon])
    return labels, rows, cols, cell_lat, cell_lon


def load_grid_data(aladin_dir, scenarios, predictors, target, gcm_grid_file):
    # Predictors upscaled onto the GCM grid, kept as maps: coarse (T, V, rows, cols), mean of the 3 km land
    # pixels inside each cell (NaN for a cell without land). Target Y (T, 1, H, W) on the 3 km grid.
    y_da = load_variable_all(aladin_dir, scenarios, target)
    lat, lon = y_da["lat"].values, y_da["lon"].values
    labels, rows, cols, cell_lat, cell_lon = gcm_grid(lat, lon, gcm_grid_file)
    gy, gx = len(cell_lat), len(cell_lon)
    land0 = np.isfinite(y_da.values[0]).ravel()
    land_pixels = np.zeros((gy, gx), dtype = int)
    for k in range(len(rows)):
        land_pixels[rows[k], cols[k]] = int((land0 & (labels.ravel() == k)).sum())

    upscaled = []                                                # one variable at a time keeps memory low
    for v in predictors:
        da = load_variable_all(aladin_dir, scenarios, v)
        up = upscale_to_gcm(da.values.astype(np.float32), labels, len(rows))      # (T, n_cells)
        grid_values = np.full((len(up), gy, gx), np.nan, dtype = np.float32)
        grid_values[:, rows, cols] = up
        upscaled.append((da["time"].values, grid_values))

    times = y_da["time"].values
    for t, _ in upscaled:
        times = np.intersect1d(times, t)                         # days present in every variable
    coarse = np.stack([g[np.searchsorted(t, times)] for t, g in upscaled], axis = 1).astype(np.float32)
    Y = y_da.sel(time = times).values[:, None].astype(np.float32)
    grid = {"cell_lat": [float(v) for v in cell_lat], "cell_lon": [float(v) for v in cell_lon],
            "land_pixels": land_pixels.tolist()}
    print(f"{'+'.join(scenarios)}: {len(times)} days ({str(times[0])[:10]} -> {str(times[-1])[:10]}) | "
          f"grid {Y.shape[2:]} | GCM grid {gy} x {gx} cells, land pixels per cell {land_pixels.tolist()}")
    return coarse, Y, times, lat, lon, grid


def flatten_grid(coarse, grid, predictors):
    # Older 1D layout: one value per variable and per GCM cell with land, cells in row-major order
    # (e.g. "tasminAdjust@-21.71N,54.84E"), as used by run1, gcm_predict.py and predictor_analysis.py.
    rr, cc = np.nonzero(np.array(grid["land_pixels"]) > 0)
    features = coarse[:, :, rr, cc].reshape(len(coarse), -1).astype(np.float32)
    names = [f"{v}@{grid['cell_lat'][r]:.2f}N,{grid['cell_lon'][c]:.2f}E" for v in predictors for r, c in zip(rr, cc)]
    return features, names


def load_data(aladin_dir, scenarios, predictors, target, gcm_grid_file):
    # Flat 1D layout (T, V * cells with land); same output as before the 2D GCM input
    coarse, Y, times, lat, lon, grid = load_grid_data(aladin_dir, scenarios, predictors, target, gcm_grid_file)
    features, names = flatten_grid(coarse, grid, predictors)
    return features, Y, times, lat, lon, names


def load_grid_cached(args):
    # Reading the NetCDF files through SSHFS is slow: the prepared arrays (GCM-grid predictors, target, DEM)
    # are saved once in --cache-dir on the local disk and reused while the inputs stay the same.
    key = json.dumps({"layout": "gcm-grid-v2", "aladin_dir": str(args.aladin_dir), "scenarios": args.scenarios,
                      "predictors": args.predictors, "target": args.target,
                      "gcm_grid_file": str(args.gcm_grid_file), "dem_path": str(args.dem_path),
                      "land_frac_path": str(args.land_frac_path), "dem": "land-only-v2"}, sort_keys = True)
    cache = Path(args.cache_dir) / f"data_{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz" if args.cache_dir else None

    if cache is not None and cache.exists():
        d = np.load(cache, allow_pickle = False)
        print(f"Loaded cached data: {cache}")
        grid = {"cell_lat": d["cell_lat"].tolist(), "cell_lon": d["cell_lon"].tolist(),
                "land_pixels": d["land_pixels"].tolist()}
        return d["coarse"], d["Y"], d["times"], d["lat"], d["lon"], grid, d["dem"]

    t0 = time.time()
    coarse, Y, times, lat, lon, grid = load_grid_data(args.aladin_dir, args.scenarios, args.predictors,
                                                      args.target, args.gcm_grid_file)
    dem = load_dem(args.dem_path, lat, lon, args.land_frac_path)
    print(f"Data read in {time.time() - t0:.0f} s")
    if cache is not None:
        cache.parent.mkdir(parents = True, exist_ok = True)
        np.savez(cache, coarse = coarse, Y = Y, times = times, lat = lat, lon = lon, dem = dem,
                 cell_lat = np.array(grid["cell_lat"]), cell_lon = np.array(grid["cell_lon"]),
                 land_pixels = np.array(grid["land_pixels"]), key = np.array(key))
        print(f"Saved cache: {cache}")
    return coarse, Y, times, lat, lon, grid, dem


def load_data_cached(args):
    # Flat 1D layout, same output as before: (features, Y, times, lat, lon, names, dem)
    coarse, Y, times, lat, lon, grid, dem = load_grid_cached(args)
    features, names = flatten_grid(coarse, grid, args.predictors)
    return features, Y, times, lat, lon, names, dem


def seasonal_features(times):
    # sin and cos of the day of the year, day 1 ... 365 (366 in leap years): one full cycle per year,
    # 31 December and 1 January end up next to each other. (T, 2), already in [-1, 1].
    t = times.astype("datetime64[D]")
    years = t.astype("datetime64[Y]").astype(int) + 1970
    doy = (t - t.astype("datetime64[Y]")).astype(int) + 1
    n_days = np.where((years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0)), 366, 365)
    angle = 2 * np.pi * doy / n_days
    return np.stack([np.sin(angle), np.cos(angle)], axis = 1).astype(np.float32)


def parse_periods(specs):
    # ["2011-2014", "2046-2050"] -> [(2011, 2014), (2046, 2050)]
    periods = []
    for p in specs:
        y0, y1 = (int(v) for v in str(p).split("-"))
        periods.append((min(y0, y1), max(y0, y1)))
    return periods


def years_of(times):
    return times.astype("datetime64[Y]").astype(int) + 1970


def split_years(years, n_folds):
    # Splits a set of years into n_folds groups of consecutive years, as equal as possible.
    # Stretches separated by a test period are split separately (no group straddles a test period);
    # the folds are shared between stretches in proportion to their length (largest remainder).
    years = np.unique(years)
    if len(years) < n_folds:
        raise ValueError(f"{len(years)} years cannot be split into {n_folds} folds")
    segments = np.split(years, np.where(np.diff(years) > 1)[0] + 1)
    quota = np.array([len(seg) for seg in segments]) * n_folds / len(years)
    counts = np.floor(quota).astype(int)
    for i in np.argsort(-(quota - counts))[:n_folds - counts.sum()]:
        counts[i] += 1
    groups = []
    for seg, c in zip(segments, counts):
        groups += [g for g in np.array_split(seg, c) if len(g)] if c > 0 else []
    if len(groups) != n_folds:               # very short stretch got no fold: fall back to one sequence
        groups = np.array_split(years, n_folds)
    return groups


def time_folds(times, test_periods, n_folds = N_FOLDS, projection_start = PROJECTION_START):
    # Fixed test set: every day inside test_periods, never used for training or early stopping.
    # k-fold cross-validation on the remaining years: the historical part (< projection_start) and the
    # projection part are each split into n_folds groups of consecutive years. The validation set of fold k
    # = historical group k + projection group k, so each validation set holds both parts in the same
    # proportion as the data (about 28 % / 72 %), and every non-test year is validated exactly once.
    # All other non-test days are the fold's training set.
    years = years_of(times)
    is_test = np.zeros(len(times), dtype = bool)
    for y0, y1 in test_periods:
        is_test |= (years >= y0) & (years <= y1)
    test = np.where(is_test)[0]

    parts = [years[~is_test & (years < projection_start)], years[~is_test & (years >= projection_start)]]
    groups = [split_years(p, n_folds) for p in parts if len(p)]  # one part only -> plain k-fold

    shares = [len(p) / (~is_test).sum() for p in parts]
    print(f"Non-test days: historical {shares[0]:.1%} | projection {shares[1]:.1%}")

    folds = []
    for k in range(n_folds):
        chosen = [g[k] for g in groups]
        in_val = np.isin(years, np.concatenate(chosen)) & ~is_test
        folds.append({"fold": k, "val_blocks": [(int(c[0]), int(c[-1])) for c in chosen],
                      "train": np.where(~is_test & ~in_val)[0], "val": np.where(in_val)[0], "test": test})
    return folds


def blocks_label(blocks, sep = "+"):
    return sep.join(f"{a}-{b}" for a, b in blocks)


def score(pred, truth, times, test_periods):
    # RMSE / MAE / bias on land pixels (NaN in truth = sea), for the whole test set and for each test period
    def metrics(d):
        return {"rmse": float(np.sqrt(np.nanmean(d ** 2))), "mae": float(np.nanmean(np.abs(d))),
                "bias": float(np.nanmean(d)), "n_scored": int(np.isfinite(d).sum())}
    diff = pred - truth
    years = years_of(times)
    out = {f"{k}_all": v for k, v in metrics(diff).items()}
    for y0, y1 in test_periods:
        sel = (years >= y0) & (years <= y1)
        out.update({f"{k}_{y0}-{y1}": v for k, v in metrics(diff[sel]).items() if k != "n_scored"})
    return out


def normalization_stats(features, Y, dem):
    return {"f_mean": np.nanmean(features, axis = 0).tolist(), "f_std": np.nanstd(features, axis = 0).tolist(),
            "y_mean": float(np.nanmean(Y)), "y_std": float(np.nanstd(Y)),
            "dem_mean": float(dem.mean()), "dem_std": float(dem.std())}


def normalize_features(features, stats):
    f_mean = np.array(stats["f_mean"], dtype = np.float32)
    f_std = np.array(stats["f_std"], dtype = np.float32) + 1e-8
    return np.nan_to_num((features - f_mean) / f_std).astype(np.float32)


def pad_to_multiple(x, multiple):
    # The contractive path halves H, W number_of_blocks times: pad so they stay even
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode = "replicate"), (h, w)


class DownscalingModel(nn.Module):
    # Wraps the UNET. The DEM is stored as a buffer (it is the same for every day) and fed as the 2D input.
    #   coarse_shape given : the GCM-grid predictors (variables x rows x cols) are a 2D input joined at the
    #                        bottleneck; n_features = length of the optional 1D vector (sin/cos of the day), 0 = none
    #   coarse_shape None  : older layout (e.g. run1): one flat 1D vector of n_features joined at the bottleneck
    # Output: (N, 1, H, W) on the 3 km grid.
    def __init__(self, static, n_features, number_of_blocks = 4, neurons = (64, 128, 256),
                 coarse_shape = None, coarse_channels = 128, **unet_kwargs) -> None:
        super().__init__()

        self.multiple = 2 ** number_of_blocks
        static = torch.as_tensor(static, dtype = torch.float32)[None]          # (1, S, H, W)
        static, self.hr_shape = pad_to_multiple(static, self.multiple)
        self.register_buffer("static", static)

        self.unet = UNET(input_shape_2D = (1, *static.shape[1:]),
                         input_shape_1D = (1, n_features) if n_features else None,
                         input_shape_coarse2D = coarse_shape, coarse_channels = coarse_channels,
                         output_channels = 1, number_of_blocks = number_of_blocks,
                         inputNN_neurons_list = list(neurons), double_resolution_times = 0,
                         **unet_kwargs)

    def forward(self, x_1D = None, x_coarse = None):
        h, w = self.hr_shape
        n = (x_coarse if x_coarse is not None else x_1D).shape[0]
        x_2D = self.static.expand(n, -1, -1, -1)
        return self.unet(x_2D, x_1D, x_coarse)[..., :h, :w]


def masked_mse(y_pred, y_true):
    mask = ~torch.isnan(y_true)
    return ((y_pred[mask] - y_true[mask]) ** 2).mean()


def batches(inputs, y, idx, batch_size, shuffle = False):
    # Mini-batches drawn from tensors that already sit on the device (the whole dataset fits in GPU memory),
    # which avoids a DataLoader copying every batch from the CPU. Training drops the last incomplete batch:
    # BatchNorm1d needs more than one sample. inputs: dict of model inputs, e.g. {"x_coarse": ..., "x_1D": ...}.
    idx = torch.as_tensor(idx, device = y.device)
    if shuffle:
        idx = idx[torch.randperm(len(idx), device = y.device)]
    stop = len(idx) - (len(idx) % batch_size) if shuffle else len(idx)
    for i in range(0, stop, batch_size):
        b = idx[i:i + batch_size]
        yield {k: t[b] for k, t in inputs.items()}, y[b]


def run_epoch(model, batch_iter, optimizer = None, amp_dtype = None, scaler = None):
    model.train(optimizer is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, y in batch_iter:
            with torch.autocast(device_type = y.device.type, dtype = amp_dtype, enabled = amp_dtype is not None):
                y_pred = model(**inputs)
            loss = masked_mse(y_pred.float(), y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none = True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += loss.item() * y.shape[0]
            n += y.shape[0]
    return total / n


def build_model(stats, n_features, dem, device):
    # Rebuilds the network described by a fold's normalization_stats.json.
    # Runs with "coarse_shape" use the 2D GCM-grid input (+ optional sin/cos vector); older runs (no
    # "coarse_shape", e.g. run1) use one flat vector of n_features, as before.
    static = ((dem - stats["dem_mean"]) / (stats["dem_std"] + 1e-8))[None]      # (1, H, W): one static channel
    common = dict(neurons = stats["dense_neurons"], input_number_kernels = stats["base_channels"],
                  dropout_rate_2D = stats["dropout"], dropout_rate_1Dinput = stats["dropout"])
    if stats.get("coarse_shape") is None:
        return DownscalingModel(static = static, n_features = n_features, **common).to(device)
    return DownscalingModel(static = static, n_features = stats.get("n_features", 0),
                            coarse_shape = stats["coarse_shape"], coarse_channels = stats["coarse_channels"],
                            **common).to(device)


def train_fold(fold, coarse, season, Y, dem, land_mask, times, lat, lon, grid, args, device):
    import xarray as xr

    val_label, k = blocks_label(fold["val_blocks"]), fold["fold"]
    out_dir = Path(args.models_dir) / f"fold_{k:02d}_val{val_label}"
    out_dir.mkdir(parents = True, exist_ok = True)
    idx_train, idx_val, idx_test = fold["train"], fold["val"], fold["test"]
    print(f"\n=== Fold {k}: validation {blocks_label(fold['val_blocks'], ' + ')} | train {len(idx_train)} days, val {len(idx_val)}, "
          f"test {len(idx_test)} (fixed) ===")
    seed_everything(args.seed + k)

    # Statistics from this fold's training days only
    # (per variable and per GCM cell; the sin/cos of the day are already in [-1, 1] and are not normalized)
    stats = normalization_stats(coarse[idx_train], Y[idx_train], dem)
    stats.update({"predictors": args.predictors, "target": args.target, "scenarios": args.scenarios,
                  "dem_path": str(args.dem_path), "land_frac_path": str(args.land_frac_path), "base_channels": args.base_channels,
                  "dense_neurons": args.dense_neurons, "dropout": args.dropout,
                  "input_layout": "gcm_grid_2d", "coarse_shape": list(coarse.shape[1:]),
                  "coarse_channels": args.coarse_channels, "seasonal": season is not None,
                  "n_features": season.shape[1] if season is not None else 0, "gcm_grid": grid, "fold": k,
                  "val_blocks": blocks_label(fold["val_blocks"]), "test_periods": [f"{a}-{b}" for a, b in args.test_periods]})
    json.dump(stats, open(out_dir / "normalization_stats.json", "w"), indent = 2)

    # Whole normalized dataset on the device once (about 160 MB)
    inputs = {"x_coarse": torch.from_numpy(normalize_features(coarse, stats)).to(device)}   # (T, V, rows, cols)
    if season is not None:
        inputs["x_1D"] = torch.from_numpy(season).to(device)                                # (T, 2) sin, cos
    y_t = torch.from_numpy((Y - stats["y_mean"]) / (stats["y_std"] + 1e-8)).to(device)   # NaN (sea) kept, masked in the loss

    model = build_model(stats, stats["n_features"], dem, device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 7)
    amp_dtype, scaler = amp_settings(args.amp, device)

    best_val, epochs_without_improvement = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, batches(inputs, y_t, idx_train, args.batch_size, shuffle = True),
                               optimizer, amp_dtype, scaler)
        val_loss = run_epoch(model, batches(inputs, y_t, idx_val, args.eval_batch_size), amp_dtype = amp_dtype)
        scheduler.step(val_loss)
        print(f"Fold {k} | epoch {epoch:3d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f} | {time.time() - t0:.0f} s")

        if val_loss < best_val:
            best_val, epochs_without_improvement = val_loss, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping")
                break

    # Test block: predictions in target units, scored on land pixels
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location = device, weights_only = True))
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in batches(inputs, y_t, idx_test, args.eval_batch_size):
            preds.append(model(**x).float().cpu().numpy())
    pred = np.concatenate(preds)[:, 0] * stats["y_std"] + stats["y_mean"]
    pred = np.where(land_mask[None], pred, np.nan).astype(np.float32)

    metrics = {"fold": k, "val_blocks": val_label, "n_train": len(idx_train), "n_val": len(idx_val),
               "n_test": len(idx_test), "best_val_mse_norm": best_val}
    metrics.update(score(pred, Y[idx_test, 0], times[idx_test], args.test_periods))
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent = 2)

    xr.DataArray(pred, dims = ("time", "lat", "lon"), coords = {"time": times[idx_test], "lat": lat, "lon": lon},
                 name = args.target).to_netcdf(out_dir / f"{args.target}_test_pred.nc")
    print(f"Fold {k} | test set: RMSE {metrics['rmse_all']:.3f} | MAE {metrics['mae_all']:.3f} | "
          f"bias {metrics['bias_all']:+.3f} | " +
          " | ".join(f"{a}-{b} RMSE {metrics[f'rmse_{a}-{b}']:.3f}" for a, b in args.test_periods))
    del inputs, y_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def write_cv_summary(models_dir, target, Y_test, test_times, test_periods):
    # cv_summary.csv: the metrics of every fold trained so far in models_dir (all scored on the same test set),
    # plus the ensemble: the mean of the folds' test predictions, saved and scored as well.
    import xarray as xr

    models_dir = Path(models_dir)
    rows = [json.load(open(p)) for p in sorted(models_dir.glob("fold_*/metrics.json"))]
    rows = [r for r in rows if "rmse_all" in r and "val_blocks" in r]       # ignore folds from older setups
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(models_dir / "cv_summary.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in keys) + "\n")
    rmse = np.array([r["rmse_all"] for r in rows])
    print(f"\nCV summary ({len(rows)} folds, same test set): test RMSE mean {rmse.mean():.3f} +/- {rmse.std():.3f} "
          f"(min {rmse.min():.3f}, max {rmse.max():.3f}) | {models_dir / 'cv_summary.csv'}")

    preds = []
    for p in sorted(models_dir.glob(f"fold_*/{target}_test_pred.nc")):
        with xr.open_dataarray(p) as da:
            if not np.array_equal(da["time"].values, test_times):
                print(f"Skipped in the ensemble (different test days, older run?): {p}")
                continue
            preds.append(da.values)
    if not preds:
        return
    ensemble = np.mean(preds, axis = 0).astype(np.float32)
    ens_metrics = {"n_folds": len(preds), **score(ensemble, Y_test, test_times, test_periods)}
    json.dump(ens_metrics, open(models_dir / "ensemble_metrics.json", "w"), indent = 2)
    with xr.open_dataarray(sorted(models_dir.glob(f"fold_*/{target}_test_pred.nc"))[0]) as ref:
        ref.copy(data = ensemble).to_netcdf(models_dir / f"{target}_test_pred_ensemble.nc")
    print(f"Ensemble of {len(preds)} folds: test RMSE {ens_metrics['rmse_all']:.3f} | MAE {ens_metrics['mae_all']:.3f} | "
          f"bias {ens_metrics['bias_all']:+.3f} | " +
          " | ".join(f"{a}-{b} RMSE {ens_metrics[f'rmse_{a}-{b}']:.3f}" for a, b in test_periods))


# ---------------------------------------------------------------- machine settings

def get_device(name):
    # "auto" -> first CUDA GPU, then Apple MPS, then CPU. Also accepts "cuda", "cuda:1", "mps", "cpu".
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {name} requested but CUDA is not available (torch {torch.__version__}, "
                           f"built for CUDA {torch.version.cuda})")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"--device {name}: only {torch.cuda.device_count()} GPU(s) visible")
    return device


def amp_settings(enabled, device):
    # Mixed precision on GPU: bfloat16 when supported (no gradient scaling needed), else float16 with a GradScaler
    if not enabled or device.type != "cuda":
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds every CUDA device


def resolve_paths(args):
    data_dir = Path(args.data_dir).expanduser()
    args.data_dir = data_dir
    args.aladin_dir = Path(args.aladin_dir).expanduser() if args.aladin_dir else data_dir / ALADIN_SUBDIR
    args.dem_path = Path(args.dem_path).expanduser() if args.dem_path else data_dir / DEM_SUBPATH
    land_frac = getattr(args, "land_frac_path", None)
    args.land_frac_path = Path(land_frac).expanduser() if land_frac else data_dir / LAND_FRAC_SUBPATH
    args.gcm_grid_file = Path(args.gcm_grid_file).expanduser() if args.gcm_grid_file else data_dir / GCM_GRID_SUBPATH
    if not args.gcm_grid_file.exists():                          # file moved inside the GCM folder? look for it by name
        found = sorted((data_dir / Path(GCM_GRID_SUBPATH).parts[0]).glob(f"**/{args.gcm_grid_file.name}"))
        if found:
            print(f"GCM grid file not at {args.gcm_grid_file}, using {found[0]}")
            args.gcm_grid_file = found[0]
    args.models_dir = Path(args.models_dir).expanduser()
    args.cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None

    missing = [f"{name}: {p}" for name, p in (("data dir", data_dir), ("ALADIN dir", args.aladin_dir),
                                               ("DEM", args.dem_path), ("land fraction", args.land_frac_path),
                                               ("GCM grid file", args.gcm_grid_file))
               if not p.exists()]
    if missing:
        raise FileNotFoundError("Not found (is the NAS mounted? see --data-dir):\n  " + "\n  ".join(missing))
    return args


def drop_target_from_predictors(predictors, target):
    # The target is never its own predictor (it would leak the answer into the input).
    kept = [v for v in predictors if v != target]
    if len(kept) < len(predictors):
        print(f"Target {target} removed from the predictors -> {', '.join(kept)}")
    if not kept:
        raise ValueError(f"No predictor left once the target {target} is removed: pass --predictors")
    return kept


def train(args):
    args = resolve_paths(args)
    if args.keep_target_predictor:
        if args.target in args.predictors:
            print(f"WARNING: target {args.target} kept among the predictors (--keep-target-predictor): "
                  "the input contains the answer at GCM scale, so scores will be optimistic")
    else:
        args.predictors = drop_target_from_predictors(args.predictors, args.target)
    if args.seed < 0:                                            # fresh random seed for this run, recorded in run_config.json
        args.seed = int.from_bytes(os.urandom(4), "little") % 2**31
    device = get_device(args.device)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True                   # fixed input size: let cuDNN pick the fastest kernels
    args.models_dir.mkdir(parents = True, exist_ok = True)

    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    print(f"Host {platform.node()} | torch {torch.__version__} | device {device}{gpu} | CPU threads {torch.get_num_threads()}"
          f" | AMP {'on' if amp_settings(args.amp, device)[0] is not None else 'off'} | seed {args.seed}")
    print(f"Data {args.data_dir} | models {args.models_dir} | cache {args.cache_dir or 'off'}")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update({"host": platform.node(), "torch": torch.__version__, "device_used": str(device) + gpu,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    json.dump(config, open(args.models_dir / "run_config.json", "w"), indent = 2)

    coarse, Y, times, lat, lon, grid, dem = load_grid_cached(args)
    season = seasonal_features(times) if args.seasonal else None
    land_mask = ~np.isnan(Y[0, 0])
    np.save(args.models_dir / "land_mask.npy", land_mask)
    print(f"DEM {dem.shape}: {dem.min():.0f}-{dem.max():.0f} m | land pixels: {land_mask.sum()}/{land_mask.size}")
    print(f"Inputs: GCM grid {coarse.shape[1]} variables x {coarse.shape[2]} x {coarse.shape[3]} cells (2D, at the bottleneck)"
          f" | sin/cos day of year: {'on' if season is not None else 'off'}")

    args.test_periods = parse_periods(args.test_periods)
    folds = time_folds(times, args.test_periods, args.n_folds, args.projection_start)
    if len(folds[0]["test"]) == 0:
        raise ValueError(f"No days inside the test periods {args.test_periods}")
    print(f"Test set (fixed): {', '.join(f'{a}-{b}' for a, b in args.test_periods)} -> {len(folds[0]['test'])} days")
    print("Folds (validation = historical years + projection years):\n  " +
          "\n  ".join(f"{f['fold']:2d}: {blocks_label(f['val_blocks'], ' + ')}" for f in folds))
    if args.dry_run:
        print("Dry run: data and folds OK, no training.")
        return
    selected = folds if args.folds is None else [folds[k] for k in args.folds]

    for fold in selected:
        train_fold(fold, coarse, season, Y, dem, land_mask, times, lat, lon, grid, args, device)
    test = folds[0]["test"]
    write_cv_summary(args.models_dir, args.target, Y[test, 0], times[test], args.test_periods)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering = True)              # write each line to train.log at once under nohup
    sys.stderr.reconfigure(line_buffering = True)
    env = os.environ.get
    parser = argparse.ArgumentParser(description = "UNET: GCM-grid (1.4 deg) predictors + DEM -> 3 km, blocked time cross-validation",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    paths = parser.add_argument_group("paths (flag > environment variable > default)")
    paths.add_argument("--data-dir", default = env("UNET_DATA_DIR", DEFAULT_DATA_DIR), help = "NAS data root, e.g. the SSHFS mount point [UNET_DATA_DIR]")
    paths.add_argument("--aladin-dir", default = None, help = f"default: <data-dir>/{ALADIN_SUBDIR}")
    paths.add_argument("--dem-path", default = None, help = f"default: <data-dir>/{DEM_SUBPATH}")
    paths.add_argument("--land-frac-path", default = None, help = f"fracLand mask, sea DEM -> 0 m; default: <data-dir>/{LAND_FRAC_SUBPATH}")
    paths.add_argument("--gcm-grid-file", default = None, help = f"default: <data-dir>/{GCM_GRID_SUBPATH}")
    paths.add_argument("--models-dir", default = env("UNET_MODELS_DIR", DEFAULT_MODELS_DIR), help = "outputs; prefer a local disk [UNET_MODELS_DIR]")
    paths.add_argument("--cache-dir", default = env("UNET_CACHE_DIR", DEFAULT_CACHE_DIR), help = "local cache of the prepared arrays, '' = off [UNET_CACHE_DIR]")

    machine = parser.add_argument_group("machine")
    machine.add_argument("--device", default = env("UNET_DEVICE", DEFAULT_DEVICE), help = "auto, cuda, cuda:1, mps or cpu [UNET_DEVICE]")
    machine.add_argument("--threads", type = int, default = int(env("UNET_THREADS", "0")), help = "CPU threads for torch, 0 = torch default [UNET_THREADS]")
    machine.add_argument("--amp", action = "store_true", help = "mixed precision on GPU (faster, bfloat16 when supported)")
    machine.add_argument("--seed", type = int, default = 42,
                         help = "seeds the shuffling of the training days (new order every epoch) and the weights; "
                                "-1 = different random seed for every run")
    machine.add_argument("--dry-run", action = "store_true", help = "check paths, read data and list folds, then stop")

    exp = parser.add_argument_group("experiment")
    exp.add_argument("--scenarios", nargs = "+", default = SCENARIOS, help = "joined in time for training")
    exp.add_argument("--test-periods", nargs = "+", default = TEST_PERIODS, help = "fixed test set, years as START-END")
    exp.add_argument("--n-folds", type = int, default = N_FOLDS, help = "number of cross-validation folds")
    exp.add_argument("--projection-start", type = int, default = PROJECTION_START,
                     help = "first projection year; each validation set takes years before and after it")
    exp.add_argument("--folds", type = int, nargs = "+", default = None, help = "fold indices to run (default: all)")
    exp.add_argument("--predictors", nargs = "+", default = PREDICTORS, help = "the target is removed from this list if present")
    exp.add_argument("--target", default = TARGET)
    exp.add_argument("--keep-target-predictor", action = "store_true",
                     help = "keep the target among the predictors if listed (leakage: optimistic scores)")

    model = parser.add_argument_group("model and training")
    model.add_argument("--base-channels", type = int, default = 32)
    model.add_argument("--dense-neurons", type = int, nargs = "+", default = [32, 64, 128],
                       help = "InputBlock1D layers for the sin/cos of the day")
    model.add_argument("--coarse-channels", type = int, default = 128,
                       help = "channels the GCM-grid predictors are projected to at the bottleneck")
    model.add_argument("--no-seasonal", dest = "seasonal", action = "store_false",
                       help = "leave out the sin/cos of the day of the year")
    model.add_argument("--dropout", type = float, default = 0.)
    model.add_argument("--batch-size", type = int, default = 16)
    model.add_argument("--eval-batch-size", type = int, default = 256)
    model.add_argument("--lr", type = float, default = 1e-4)
    model.add_argument("--epochs", type = int, default = 75)
    model.add_argument("--patience", type = int, default = 15)
    train(parser.parse_args())
