"""
UNET.py - U-Net statistical downscaling (PyTorch)
=================================================

Purpose
-------
Downscale daily climate variables over La Reunion from the CNRM-ESM2-1 GCM grid (~1.4 deg)
to 3 km, using the bias-corrected ALADIN data (ALADIN_CORRIGE) and a DEM.

Inputs / output
---------------
  1D input : each predictor upscaled onto the GCM grid: mean of the 3 km land pixels inside
             each 1.4 deg cell. The domain overlaps 2 x 2 GCM cells, so 6 variables x 4 cells
             = 24 numbers per day. Goes through InputBlock1D (dense stack) and is injected
             at the bottleneck.
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

Training and evaluation
-----------------------
  historical (1981-2014) and ssp370 (2015-2100) are joined into one daily series.
  Blocked time cross-validation: the series is cut into consecutive 5-year blocks
  (1981-1985, ..., 2096-2100: 24 folds). In each fold one block is the test set, the
  block before it (after it for the first fold) is the validation set for early stopping,
  and all other days are training data. Normalization statistics come from the fold's
  training days only. Loss: masked MSE on land pixels.

Data (under DATA_DIR = /mnt/nas/Data)
-------------------------------------
  METEO_FRANCE/DONNEES_SIMULATIONS/ALADIN/ALADIN_CORRIGE/{historical,ssp370}/day/{var}/*.nc
  CMIP_CNRM_ESM21/tas_day_*historical*.nc   (only its lat/lon: the GCM grid)
  DEM/gebco_2025_*.nc                       (DEM, variable "elevation")
  Predictors: tasmin, hursmax, hursmin, prtot, rsds, sfcWind (the *Adjust variables);
  default target: tasmaxAdjust (not among the predictors).

Usage
-----
  python UNET.py                          # all 24 folds
  python UNET.py --folds 0 23             # only some folds (indices printed at start)
  python UNET.py --test-years 10 --target tasminAdjust --batch-size 64

Outputs (models/unet_3km/)
--------------------------
  land_mask.npy, cv_summary.csv (all folds trained so far, pooled RMSE printed)
  fold_XX_YYYY-YYYY/: best.pt, normalization_stats.json, metrics.json,
                      {target}_pred_YYYY-YYYY.nc (out-of-fold predictions for the test block)

Known limitations
-----------------
  - Upscaled ALADIN is a stand-in for the GCM: real CNRM-ESM2-1 cells also include ocean.
  - ssp370 files mix ALADIN63 and ALADIN64 runs (see file names).
  - Neighbouring blocks share a boundary: the last days of one block and the first days of
    the next are correlated (weather persists a few days). Small effect with 5-year blocks.
  - ContractiveBlock loop in UNET.__init__: block 0 is built then overwritten by the
    `else` branch, so it does not use replicate padding.
  - DoubleResolutionLayer ignores `output_padding`; ExpansiveBlock does not pass
    `dropout_rate` to it.
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
# Training: GCM-grid (1.4 deg) predictors + DEM  ->  3 km target
#   2D input : static DEM on the 3 km grid  (InputBlock2D + skip connections)
#   1D input : predictors upscaled onto the CNRM-ESM2-1 grid (~1.4 deg), one value per variable
#              and per GCM cell overlapping the domain, flattened (InputBlock1D -> bottleneck)
#   output   : target variable on the 3 km grid (ALADIN_CORRIGE)
# Data: historical (1981-2014) + ssp370 (2015-2100), joined in time.
# Evaluation: blocked time cross-validation, one 5-year test block per fold.
# =====================================================================

DATA_DIR = Path("/mnt/nas/Data")
ALADIN_DIR = DATA_DIR / "METEO_FRANCE" / "DONNEES_SIMULATIONS" / "ALADIN" / "ALADIN_CORRIGE"   # {scenario}/day/{var}/*.nc
SCENARIOS = ["historical", "ssp370"]
PREDICTORS = ["tasminAdjust", "hursmaxAdjust", "hursminAdjust",
              "prtotAdjust", "rsdsAdjust", "sfcWindAdjust"]       # present in both historical and ssp370; target excluded
TARGET = "tasmaxAdjust"
DEM_PATH = DATA_DIR / "DEM" / "gebco_2025_n-20.82_s-21.45_w55.15_e55.9.nc"
GCM_GRID_FILE = DATA_DIR / "CMIP_CNRM_ESM21" / "tas_day_CNRM-ESM2-1_historical_r1i1p1f2_gr_18500101-20141231.nc"
TEST_YEARS = 5
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


def load_variable_all(scenarios, var):
    import xarray as xr

    da = xr.concat([load_variable(s, var) for s in scenarios], dim = "time")
    _, keep = np.unique(da["time"].values, return_index = True)   # drop duplicated dates at scenario joins, if any
    return da.isel(time = np.sort(keep))


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


def gcm_cells(lat, lon, gcm_grid_file = GCM_GRID_FILE):
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


def load_data(scenarios, predictors, target, gcm_grid_file = GCM_GRID_FILE):
    y_da = load_variable_all(scenarios, target)
    lat, lon = y_da["lat"].values, y_da["lon"].values
    labels, centres = gcm_cells(lat, lon, gcm_grid_file)
    first = y_da.values[0].ravel()
    land_cells = [k for k in range(len(centres)) if np.isfinite(first[labels.ravel() == k]).any()]

    upscaled = []                                                # one variable at a time keeps memory low
    for v in predictors:
        da = load_variable_all(scenarios, v)
        up = upscale_to_gcm(da.values.astype(np.float32), labels, len(centres))[:, land_cells]
        upscaled.append((da["time"].values, up, v))

    times = y_da["time"].values
    for t, _, _ in upscaled:
        times = np.intersect1d(times, t)                         # days present in every variable

    cols, names = [], []
    for t, up, v in upscaled:
        cols.append(up[np.searchsorted(t, times)])
        names += [f"{v}@{centres[k][0]:.2f}N,{centres[k][1]:.2f}E" for k in land_cells]
    features = np.concatenate(cols, axis = 1).astype(np.float32)                 # (T, V * cells)
    Y = y_da.sel(time = times).values[:, None].astype(np.float32)               # (T, 1, H, W)

    print(f"{'+'.join(scenarios)}: {len(times)} days ({str(times[0])[:10]} -> {str(times[-1])[:10]}) | "
          f"grid {Y.shape[2:]} | GCM cells with land: {len(land_cells)} -> 1D features {features.shape[1]}")
    return features, Y, times, lat, lon, names


def time_folds(times, test_years = TEST_YEARS):
    # Blocked time cross-validation: consecutive test_years blocks; each block is the test set once,
    # the block before it (after it, for the first) is the validation set for early stopping, the rest is training.
    years = times.astype("datetime64[Y]").astype(int) + 1970
    starts = np.arange(years.min(), years.max() + 1, test_years)
    blocks = [np.where((years >= s) & (years < s + test_years))[0] for s in starts]
    folds = []
    for k, test in enumerate(blocks):
        val = blocks[k - 1] if k > 0 else blocks[k + 1]
        train = np.setdiff1d(np.arange(len(times)), np.concatenate([test, val]))
        folds.append({"fold": k, "years": (int(years[test].min()), int(years[test].max())),
                      "train": train, "val": val, "test": test})
    return folds


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


def train_fold(fold, features, Y, dem, land_mask, times, lat, lon, names, args, device):
    from torch.utils.data import DataLoader, TensorDataset
    import xarray as xr

    (y0, y1), k = fold["years"], fold["fold"]
    out_dir = MODELS_DIR / f"fold_{k:02d}_{y0}-{y1}"
    out_dir.mkdir(parents = True, exist_ok = True)
    idx_train, idx_val, idx_test = fold["train"], fold["val"], fold["test"]
    print(f"\n=== Fold {k}: test {y0}-{y1} | train {len(idx_train)} days, val {len(idx_val)}, test {len(idx_test)} ===")

    # Statistics from this fold's training days only
    stats = normalization_stats(features[idx_train], Y[idx_train], dem)
    stats.update({"predictors": args.predictors, "target": args.target, "scenarios": args.scenarios,
                  "dem_path": str(args.dem_path), "base_channels": args.base_channels,
                  "dense_neurons": args.dense_neurons, "dropout": args.dropout,
                  "n_features": features.shape[1], "feature_names": names, "fold": k, "test_years": [y0, y1]})
    json.dump(stats, open(out_dir / "normalization_stats.json", "w"), indent = 2)

    f_norm = normalize_features(features, stats)
    Y_norm = (Y - stats["y_mean"]) / (stats["y_std"] + 1e-8)     # NaN (sea) kept, masked in the loss

    def loader(idx, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(f_norm[idx]), torch.from_numpy(Y_norm[idx])),
                          batch_size = args.batch_size, shuffle = shuffle, drop_last = shuffle)   # BatchNorm1d needs > 1 sample
    train_loader, val_loader = loader(idx_train, True), loader(idx_val, False)

    model = build_model(stats, features.shape[1], dem, device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 7)

    best_val, epochs_without_improvement = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        scheduler.step(val_loss)
        print(f"Fold {k} | epoch {epoch:3d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

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
        for i in range(0, len(idx_test), 256):
            preds.append(model(torch.from_numpy(f_norm[idx_test[i:i + 256]]).to(device)).cpu().numpy())
    pred = np.concatenate(preds)[:, 0] * stats["y_std"] + stats["y_mean"]
    pred = np.where(land_mask[None], pred, np.nan).astype(np.float32)

    diff = pred - Y[idx_test, 0]
    metrics = {"fold": k, "test_start": y0, "test_end": y1, "n_train": len(idx_train), "n_test": len(idx_test),
               "best_val_mse_norm": best_val, "test_rmse": float(np.sqrt(np.nanmean(diff ** 2))),
               "test_mae": float(np.nanmean(np.abs(diff))), "test_bias": float(np.nanmean(diff)),
               "n_scored": int(np.isfinite(diff).sum())}
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent = 2)

    xr.DataArray(pred, dims = ("time", "lat", "lon"), coords = {"time": times[idx_test], "lat": lat, "lon": lon},
                 name = args.target).to_netcdf(out_dir / f"{args.target}_pred_{y0}-{y1}.nc")
    print(f"Fold {k} | test {y0}-{y1}: RMSE {metrics['test_rmse']:.3f} | MAE {metrics['test_mae']:.3f} | "
          f"bias {metrics['test_bias']:+.3f} (target units)")
    return metrics


def write_cv_summary():
    # Collects metrics.json of every fold trained so far into cv_summary.csv
    rows = [json.load(open(p)) for p in sorted(MODELS_DIR.glob("fold_*/metrics.json"))]
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(MODELS_DIR / "cv_summary.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in keys) + "\n")
    n = np.array([r["n_scored"] for r in rows])
    pooled_rmse = np.sqrt(np.sum(n * np.array([r["test_rmse"] for r in rows]) ** 2) / n.sum())
    print(f"\nCV summary ({len(rows)} folds): pooled test RMSE {pooled_rmse:.3f} | "
          f"per fold min {min(r['test_rmse'] for r in rows):.3f}, max {max(r['test_rmse'] for r in rows):.3f} "
          f"| {MODELS_DIR / 'cv_summary.csv'}")


def train(args):
    MODELS_DIR.mkdir(parents = True, exist_ok = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, Y, times, lat, lon, names = load_data(args.scenarios, args.predictors, args.target)
    dem = load_dem(args.dem_path, lat, lon)
    land_mask = ~np.isnan(Y[0, 0])
    np.save(MODELS_DIR / "land_mask.npy", land_mask)
    print(f"DEM {dem.shape}: {dem.min():.0f}-{dem.max():.0f} m | land pixels: {land_mask.sum()}/{land_mask.size} | device: {device}")

    folds = time_folds(times, args.test_years)
    print("Folds: " + ", ".join(f"{f['fold']}:{f['years'][0]}-{f['years'][1]}" for f in folds))
    selected = folds if args.folds is None else [folds[k] for k in args.folds]

    for fold in selected:
        train_fold(fold, features, Y, dem, land_mask, times, lat, lon, names, args, device)
    write_cv_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "UNET: GCM-grid (1.4 deg) predictors + DEM -> 3 km, blocked time cross-validation")
    parser.add_argument("--scenarios", nargs = "+", default = SCENARIOS, help = "joined in time for training")
    parser.add_argument("--test-years", type = int, default = TEST_YEARS, help = "length of each test block")
    parser.add_argument("--folds", type = int, nargs = "+", default = None, help = "fold indices to run (default: all)")
    parser.add_argument("--predictors", nargs = "+", default = PREDICTORS)
    parser.add_argument("--target", default = TARGET)
    parser.add_argument("--dem-path", default = DEM_PATH)
    parser.add_argument("--base-channels", type = int, default = 32)
    parser.add_argument("--dense-neurons", type = int, nargs = "+", default = [64, 128, 256], help = "InputBlock1D layers")
    parser.add_argument("--dropout", type = float, default = 0.)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--epochs", type = int, default = 75)
    parser.add_argument("--patience", type = int, default = 15)
    train(parser.parse_args())
