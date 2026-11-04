"""
UNET.py - U-Net statistical downscaling (PyTorch)
=================================================

Purpose
-------
Downscale daily climate variables over La Reunion from ~150 km (GCM-like) resolution
to 3 km, using the bias-corrected ALADIN data (ALADIN_CORRIGE) and a DEM.

Inputs / output
---------------
  1D input : predictors upscaled to ~COARSE_KM (block mean, sea ignored) and flattened
             to a vector (T, V * gh * gw). Domain < 150 km  ->  one value per variable.
             Goes through InputBlock1D (dense stack) and is injected at the bottleneck.
  2D input : static DEM (GEBCO, averaged onto the 3 km grid, sea = 0 m), one channel.
             Goes through InputBlock2D and the skip connections.
  output   : target variable on the 3 km grid, (N, 1, H, W).

Architecture (default: 32 base channels, 4 blocks)
--------------------------------------------------
  DEM -> InputBlock2D (32) -> 4 x ContractiveBlock (MaxPool + 2 x Conv3x3-BN-ReLU)
                                 channels 64 / 128 / 256 / 256, resolution H/2 ... H/16
  vector -> InputBlock1D -> BottleNeckBlock (conv to 1x1, concat vector, ConvTranspose back)
  -> 4 x ExpansiveBlock (bilinear Upsample x2 -> Conv3x3 -> concat skip -> 2 x Conv3x3)
  -> OutputBlock (1x1 Conv; optional extra resolution doublings, 0 here)

  Building blocks (bottom -> top):
    DenseLayer, ConvolutionalLayer, ConvolutionalBlock, ContractiveBlock,
    DoubleResolutionLayer, ConcatChannelsLayer, ExpansiveBlock, BottleNeckBlock,
    InputBlock2D, InputBlock1D, OutputBlock, UNET.
  UNET.forward(x_2D, x_1D). Without input_shape_1D it is a plain U-Net.
  H and W must be multiples of 2**number_of_blocks: DownscalingModel pads the static
  DEM and crops the output back to the 3 km grid.

Pipeline
--------
  load_data -> coarsen_to_vector -> normalization (train-split stats) -> DownscalingModel
  (DEM stored as a buffer, same for every day) -> masked MSE on land pixels.

Data (under DATA_DIR = /mnt/nas/Data)
-------------------------------------
  METEO_FRANCE/DONNEES_SIMULATIONS/ALADIN/ALADIN_CORRIGE/{historical,ssp370}/day/{var}/*.nc
  1D_data/gebco_2025_*.nc   (DEM, variable "elevation")
  Predictors: tasmax, tasmin, hursmax, hursmin, prtot, rsds, sfcWind (the *Adjust variables);
  default target: tasmaxAdjust. historical covers 1981-2014, ssp370 covers 2015-2100.

Usage
-----
  python UNET.py                          # train on historical (chronological 70/15/15 split)
  python UNET.py --mode predict           # apply the saved model to ssp370
  python UNET.py --target tasminAdjust --epochs 75 --base-channels 32 --coarse-km 150

Outputs (models/unet_3km/)
--------------------------
  best.pt, normalization_stats.json, splits.npz, land_mask.npy,
  {target}_{scenario}_3km_pred.nc  (from --mode predict)

Known limitations
-----------------
  - ContractiveBlock loop in UNET.__init__: block 0 is built then overwritten by the
    `else` branch, so it does not use replicate padding.
  - DoubleResolutionLayer ignores `output_padding`; ExpansiveBlock does not pass
    `dropout_rate` to it.
  - ssp370 goes beyond the 1981-2014 training range: check extrapolation.
"""

import argparse
import glob
import json
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
                                         multiplies_channels_by = (2 if self.input_shape_1D == None else 1),
                                         padding_mode = "replicate")                
            if i == (number_of_blocks-1):
                block = ContractiveBlock( in_channels, kernel_size = kernel_size, dropout_rate = dropout_rate_2D, 
                                         multiplies_channels_by = (2 if self.input_shape_1D == None else 1) )
                #last block is not connected with skip, 
                # also in orignal model description last block does NOT double number of features, because the bottleneck adds them
                # here if 1D input is set to None the channels will be multiplied.
            else: 
                block = ContractiveBlock(in_channels, kernel_size = kernel_size, dropout_rate = dropout_rate_2D, multiplies_channels_by = 2)
                skip_x_out_channels_list.append(block.out_channels)

            in_channels = block.out_channels
            self.contractivePath.append( block )


        
        if self.input_shape_1D != None:
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





    def forward(self, x, x_1D = None):  # fixme 1d input
        
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

        if self.input_block1D != None:
            x = self.bottleneck_block(x = x, input1Dblock_output = x_1D)
            #print("after bottleneck_block shape: ", x.shape)

        for expasive_block, skip_x in zip(self.expansivePath, skip_x_list[::-1]): 
            x = expasive_block(x = x, x_from_skip_connection = skip_x)
            #print("after expansivePath block shape: ", x.shape)
            
        x = self.output_block(x)
        #print("after output_block shape: ", x.shape)
        return(x)





# =====================================================================
# Training: coarse (~150 km, GCM-like) predictors + DEM  ->  3 km target
#   2D input : static DEM on the 3 km grid  (InputBlock2D + skip connections)
#   1D input : predictors upscaled to ~COARSE_KM, flattened to a vector (InputBlock1D -> bottleneck)
#   output   : target variable on the 3 km grid (ALADIN_CORRIGE)
# Train on `historical`, apply to `ssp370` with --mode predict.
# =====================================================================

DATA_DIR = Path("/mnt/nas/Data")
ALADIN_DIR = DATA_DIR / "METEO_FRANCE" / "DONNEES_SIMULATIONS" / "ALADIN" / "ALADIN_CORRIGE"   # {historical,ssp370}/day/{var}/*.nc
PREDICTORS = ["tasmaxAdjust", "tasminAdjust", "hursmaxAdjust", "hursminAdjust",
              "prtotAdjust", "rsdsAdjust", "sfcWindAdjust"]       # present in both historical and ssp370
TARGET = "tasmaxAdjust"
DEM_PATH = DATA_DIR / "1D_data" / "gebco_2025_n-20.82_s-21.45_w55.15_e55.9.nc"
COARSE_KM = 150
MODELS_DIR = Path("models") / "unet_3km"


def load_variable(scenario, var):
    import xarray as xr

    files = sorted(glob.glob(str(ALADIN_DIR / scenario / "day" / var / "*.nc")))
    if not files:
        raise FileNotFoundError(f"No files for '{var}' in {ALADIN_DIR / scenario / 'day' / var}")

    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            parts.append(ds[var].load())
    da = xr.concat(parts, dim = "time") if len(parts) > 1 else parts[0]
    da = da.assign_coords(time = da["time"].dt.floor("D"))      # daily data: align on dates only
    return da.transpose("time", "lat", "lon")


def load_dem(path, lat, lon):
    # Averages the ~450 m GEBCO elevation over one 3 km cell, samples it at the grid centres,
    # and puts sea (<= 0 m) and cells outside the DEM extent at 0 m.
    import xarray as xr

    with xr.open_dataset(path) as ds:
        dem = ds["elevation"].load()
    window = {d: max(1, int(round(abs(float(c[1] - c[0])) / abs(float(dem[d][1] - dem[d][0])))))
              for d, c in (("lat", lat), ("lon", lon))}
    dem = dem.rolling(window, center = True, min_periods = 1).mean()
    dem = dem.interp(lat = lat, lon = lon).clip(min = 0).fillna(0)
    return dem.values.astype(np.float32)                         # (H, W)


def coarsen_to_vector(X, lat, lon, coarse_km = COARSE_KM, chunk = 2000):
    # (T, V, H, W) 3 km fields -> (T, V * gh * gw): block mean (NaN-aware, sea ignored) over ~coarse_km cells.
    # A cell never exceeds the domain, so a domain smaller than coarse_km (e.g. La Reunion) gives one value per variable.
    T, V, H, W = X.shape
    km_y = abs(float(lat[1] - lat[0])) * 111.2
    km_x = abs(float(lon[1] - lon[0])) * 111.2 * np.cos(np.deg2rad(float(np.mean(lat))))
    fy, fx = min(H, max(1, round(coarse_km / km_y))), min(W, max(1, round(coarse_km / km_x)))
    gh, gw = -(-H // fy), -(-W // fx)

    out = np.empty((T, V * gh * gw), dtype = np.float32)
    for t0 in range(0, T, chunk):
        block = np.full((min(chunk, T - t0), V, gh * fy, gw * fx), np.nan, dtype = np.float32)
        block[..., :H, :W] = X[t0:t0 + chunk]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)          # all-NaN cells -> NaN
            coarse = np.nanmean(block.reshape(len(block), V, gh, fy, gw, fx), axis = (3, 5))
        out[t0:t0 + chunk] = coarse.reshape(len(block), -1)
    return out


def load_data(scenario, predictors, target = None, coarse_km = COARSE_KM):
    import xarray as xr

    das = [load_variable(scenario, v) for v in predictors]
    if target is not None:
        das.append(load_variable(scenario, target))
    das = xr.align(*das, join = "inner")                         # common days (same lat/lon on every file)
    lat, lon = das[0]["lat"].values, das[0]["lon"].values

    X = np.stack([da.values for da in das[:len(predictors)]], axis = 1).astype(np.float32)     # (T, V, H, W)
    features = coarsen_to_vector(X, lat, lon, coarse_km)                                         # (T, F)
    Y = das[-1].values[:, None].astype(np.float32) if target is not None else None               # (T, 1, H, W)
    times = das[0]["time"].values
    print(f"[{scenario}] days: {len(times)} | grid {X.shape[2:]} -> 1D features {features.shape[1]} (~{coarse_km} km)")
    return features, Y, times, lat, lon


def chronological_splits(n, val_frac = 0.15, test_frac = 0.15):
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    idx = np.arange(n)
    return idx[:n - n_val - n_test], idx[n - n_val - n_test:n - n_test], idx[n - n_test:]


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
    # Wraps the UNET. The DEM is stored as a buffer (it is the same for every day) and fed as the 2D input,
    # the coarse predictor vector is the 1D input injected at the bottleneck. Output: (N, 1, H, W) on the 3 km grid.
    def __init__(self, static, n_features, number_of_blocks = 4, neurons = (64, 128, 256), **unet_kwargs) -> None:
        super().__init__()

        self.multiple = 2 ** number_of_blocks
        static = torch.as_tensor(static, dtype = torch.float32)[None]          # (1, S, H, W)
        static, self.hr_shape = pad_to_multiple(static, self.multiple)
        self.register_buffer("static", static)

        self.unet = UNET(input_shape_2D = (1, *static.shape[1:]), input_shape_1D = (1, n_features),
                         output_channels = 1, number_of_blocks = number_of_blocks,
                         inputNN_neurons_list = list(neurons), double_resolution_times = 0,
                         **unet_kwargs)

    def forward(self, x_1D):
        h, w = self.hr_shape
        x_2D = self.static.expand(x_1D.shape[0], -1, -1, -1)
        return self.unet(x_2D, x_1D)[..., :h, :w]


def masked_mse(y_pred, y_true):
    mask = ~torch.isnan(y_true)
    return ((y_pred[mask] - y_true[mask]) ** 2).mean()


def run_epoch(model, loader, device, optimizer = None):
    model.train(optimizer is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = masked_mse(model(x), y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.shape[0]
            n += x.shape[0]
    return total / n


def build_model(stats, n_features, dem, device):
    static = ((dem - stats["dem_mean"]) / (stats["dem_std"] + 1e-8))[None]      # (1, H, W): one static channel
    return DownscalingModel(static = static, n_features = n_features,
                            neurons = stats["dense_neurons"],
                            input_number_kernels = stats["base_channels"],
                            dropout_rate_2D = stats["dropout"],
                            dropout_rate_1Dinput = stats["dropout"]).to(device)


def train(args):
    from torch.utils.data import DataLoader, TensorDataset

    MODELS_DIR.mkdir(parents = True, exist_ok = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, Y, times, lat, lon = load_data("historical", args.predictors, args.target, args.coarse_km)
    dem = load_dem(args.dem_path, lat, lon)
    land_mask = ~np.isnan(Y[0, 0])
    np.save(MODELS_DIR / "land_mask.npy", land_mask)
    print(f"DEM {dem.shape}: {dem.min():.0f}-{dem.max():.0f} m | land pixels: {land_mask.sum()}/{land_mask.size}")

    idx_train, idx_val, idx_test = chronological_splits(len(features))
    np.savez(MODELS_DIR / "splits.npz", train = idx_train, val = idx_val, test = idx_test)

    stats = normalization_stats(features[idx_train], Y[idx_train], dem)
    stats.update({"predictors": args.predictors, "target": args.target, "coarse_km": args.coarse_km,
                  "dem_path": str(args.dem_path), "base_channels": args.base_channels,
                  "dense_neurons": args.dense_neurons, "dropout": args.dropout, "n_features": features.shape[1]})
    json.dump(stats, open(MODELS_DIR / "normalization_stats.json", "w"), indent = 2)

    features = normalize_features(features, stats)
    Y = (Y - stats["y_mean"]) / (stats["y_std"] + 1e-8)           # NaN (sea) kept in the target, masked in the loss

    def loader(idx, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(features[idx]), torch.from_numpy(Y[idx])),
                          batch_size = args.batch_size, shuffle = shuffle, drop_last = shuffle)   # BatchNorm1d needs > 1 sample
    train_loader, val_loader = loader(idx_train, True), loader(idx_val, False)

    model = build_model(stats, features.shape[1], dem, device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 7)
    print(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val, epochs_without_improvement = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

        if val_loss < best_val:
            best_val, epochs_without_improvement = val_loss, 0
            torch.save(model.state_dict(), MODELS_DIR / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(MODELS_DIR / "best.pt", map_location = device))
    test_loss = run_epoch(model, loader(idx_test, False), device)
    rmse = np.sqrt(test_loss) * stats["y_std"]
    print(f"Best val MSE {best_val:.4f} | test RMSE {rmse:.4f} (target units) | weights: {MODELS_DIR / 'best.pt'}")


def predict(args):
    # Applies the trained model to a scenario (default ssp370) and writes the 3 km field to NetCDF
    import xarray as xr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = json.load(open(MODELS_DIR / "normalization_stats.json"))
    land_mask = np.load(MODELS_DIR / "land_mask.npy")

    features, _, times, lat, lon = load_data(args.scenario, stats["predictors"], None, stats["coarse_km"])
    dem = load_dem(stats["dem_path"], lat, lon)
    features = normalize_features(features, stats)

    model = build_model(stats, features.shape[1], dem, device)
    model.load_state_dict(torch.load(MODELS_DIR / "best.pt", map_location = device))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(features), 256):
            preds.append(model(torch.from_numpy(features[i:i + 256]).to(device)).cpu().numpy())
    pred = np.concatenate(preds)[:, 0] * stats["y_std"] + stats["y_mean"]
    pred = np.where(land_mask[None], pred, np.nan).astype(np.float32)

    out = MODELS_DIR / f"{stats['target']}_{args.scenario}_3km_pred.nc"
    xr.DataArray(pred, dims = ("time", "lat", "lon"), coords = {"time": times, "lat": lat, "lon": lon},
                 name = stats["target"]).to_netcdf(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "UNET: coarse (~150 km) predictors + DEM -> 3 km downscaling")
    parser.add_argument("--mode", choices = ["train", "predict"], default = "train",
                        help = "train on historical, or predict a scenario with the saved model")
    parser.add_argument("--scenario", default = "ssp370", help = "scenario used by --mode predict")
    parser.add_argument("--predictors", nargs = "+", default = PREDICTORS)
    parser.add_argument("--target", default = TARGET)
    parser.add_argument("--dem-path", default = DEM_PATH)
    parser.add_argument("--coarse-km", type = float, default = COARSE_KM, help = "resolution of the 1D predictors (GCM-like)")
    parser.add_argument("--base-channels", type = int, default = 32)
    parser.add_argument("--dense-neurons", type = int, nargs = "+", default = [64, 128, 256], help = "InputBlock1D layers")
    parser.add_argument("--dropout", type = float, default = 0.)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--epochs", type = int, default = 75)
    parser.add_argument("--patience", type = int, default = 15)
    args = parser.parse_args()
    train(args) if args.mode == "train" else predict(args)
