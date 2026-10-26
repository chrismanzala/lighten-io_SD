import argparse
import glob
import json
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
            nn.Dropout1d(p = dropout_rate)
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
# Training: RCM predictors (ALADIN63) -> 3 km target
# Default: predictors already upscaled onto the 3 km grid (same grid, scale_factor = 1,
# no upsampling in the OutputBlock). For raw ~12 km predictors use --scale-factor 4
# (double_resolution_times = 2).
# =====================================================================

DATA_DIR = Path("/mnt/nas/Data")

# Predictors (upscaled to 3 km) and target (3 km). Adjust the patterns to the NAS layout.
LR_VARIABLES = ["tasAdjust", "hussAdjust", "hursAdjust", "prAdjust",
                "rsdsAdjust", "sfcWindAdjust", "tasmaxAdjust", "tasminAdjust"]
LR_PATTERN = str(DATA_DIR / "ALADIN63_upscaled_3km" / "daily" / "{var}" / "*.nc")   # placeholder: set to the upscaled files
HR_TARGET = "tas"
HR_PATTERN = str(DATA_DIR / "3km" / "{var}" / "*.nc")

OROG_PATH = None          # optional static 3 km orography file, added as an extra input channel
SCALE_FACTOR = 1          # 1 = same grid (upscaled predictors), 4 = raw 12 km -> 3 km
MODELS_DIR = Path("models") / "unet_3km"


def load_variable(pattern, var):
    import xarray as xr

    files = sorted(glob.glob(pattern.format(var = var)))
    if not files:
        raise FileNotFoundError(f"No files for '{var}' matching {pattern.format(var = var)}")

    ds = xr.open_mfdataset(files, combine = "by_coords")
    da = ds[var] if var in ds.data_vars else ds[list(ds.data_vars)[0]]
    da = da.squeeze(drop = True)
    da["time"] = da["time"].dt.floor("D")    # daily data: align on dates only
    return da.transpose("time", ...)         # (time, y, x)


def load_orography(path):
    import xarray as xr

    ds = xr.open_dataset(path)
    da = ds["orog"] if "orog" in ds.data_vars else ds[list(ds.data_vars)[0]]
    da = da.squeeze(drop = True)
    if "time" in da.dims:
        da = da.isel(time = 0, drop = True)
    return da                                # (y, x)


def check_same_grid(reference, others):
    # Same-grid training only makes sense if every field sits on exactly the target pixels
    spatial_dims = [d for d in reference.dims if d != "time"]
    for da in others:
        da_dims = [d for d in da.dims if d != "time"]
        if da.shape[-2:] != reference.shape[-2:]:
            raise ValueError(f"'{da.name}' grid {da.shape[-2:]} != target grid {reference.shape[-2:]}")
        for ref_name in list(spatial_dims) + ["lat", "lon"]:
            if ref_name in reference.coords and ref_name in da.coords:
                if not np.allclose(reference[ref_name].values, da[ref_name].values, atol = 1e-4, equal_nan = True):
                    raise ValueError(f"'{da.name}' coordinate '{ref_name}' differs from the target grid")
        if da_dims != spatial_dims:
            print(f"Warning: '{da.name}' dims {da_dims} differ from target dims {spatial_dims}, shapes match")


def load_data(lr_pattern, lr_variables, hr_pattern, hr_target, same_grid = True, orog_path = None):
    import xarray as xr

    lr = [load_variable(lr_pattern, v) for v in lr_variables]
    hr = load_variable(hr_pattern, hr_target)
    *lr, hr = xr.align(*lr, hr, join = "inner", exclude = [d for d in hr.dims if d != "time"])

    static = [load_orography(orog_path)] if orog_path else []
    if same_grid:
        check_same_grid(hr, lr + static)
    elif static and static[0].shape != lr[0].shape[-2:]:
        raise ValueError(f"Orography grid {static[0].shape} != predictor grid {lr[0].shape[-2:]}")

    channels = [da.values for da in lr] + [np.broadcast_to(da.values, lr[0].shape) for da in static]
    X = np.stack(channels, axis = 1).astype(np.float32)                  # (T, C, h, w)
    Y = hr.values[:, None].astype(np.float32)                            # (T, 1, H, W)
    print(f"Common days: {X.shape[0]} | LR {X.shape[2:]} -> HR {Y.shape[2:]}")
    return X, Y


def chronological_splits(n, val_frac = 0.15, test_frac = 0.15):
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    idx = np.arange(n)
    return idx[:n - n_val - n_test], idx[n - n_val - n_test:n - n_test], idx[n - n_test:]


def normalization_stats(X, Y):
    return {"x_mean": np.nanmean(X, axis = (0, 2, 3)).tolist(), "x_std": np.nanstd(X, axis = (0, 2, 3)).tolist(),
            "y_mean": float(np.nanmean(Y)), "y_std": float(np.nanstd(Y))}


def normalize(X, Y, stats):
    x_mean = np.array(stats["x_mean"], dtype = np.float32)[None, :, None, None]
    x_std = np.array(stats["x_std"], dtype = np.float32)[None, :, None, None] + 1e-8
    X = np.nan_to_num((X - x_mean) / x_std)                  # NaN (e.g. sea) -> 0 in the inputs
    Y = (Y - stats["y_mean"]) / (stats["y_std"] + 1e-8)       # NaN kept in target, masked in the loss
    return X, Y


def pad_to_multiple(x, multiple):
    # The contractive path halves H, W number_of_blocks times: pad so they stay even
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode = "replicate"), (h, w)


class DownscalingModel(nn.Module):
    # Wraps the UNET: pads the input, crops the (x scale_factor) output and resizes it onto the exact 3 km grid.
    # With scale_factor = 1 (upscaled predictors) the UNET keeps the input resolution.
    def __init__(self, in_channels, lr_shape, hr_shape, number_of_blocks = 4, scale_factor = SCALE_FACTOR, **unet_kwargs) -> None:
        super().__init__()

        self.multiple = 2 ** number_of_blocks
        self.scale_factor = scale_factor
        self.hr_shape = tuple(hr_shape)
        padded = [s + (-s) % self.multiple for s in lr_shape]

        self.unet = UNET(input_shape_2D = (1, in_channels, *padded), output_channels = 1,
                         number_of_blocks = number_of_blocks,
                         double_resolution_times = int(np.log2(scale_factor)),
                         **unet_kwargs)

    def forward(self, x):
        x, (h, w) = pad_to_multiple(x, self.multiple)
        x = self.unet(x)[..., :h * self.scale_factor, :w * self.scale_factor]
        if x.shape[-2:] != self.hr_shape:
            x = F.interpolate(x, size = self.hr_shape, mode = "bilinear", align_corners = False)
        return x


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


def train(args):
    from torch.utils.data import DataLoader, TensorDataset

    MODELS_DIR.mkdir(parents = True, exist_ok = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y = load_data(args.lr_pattern, args.lr_variables, args.hr_pattern, args.hr_target,
                     same_grid = args.scale_factor == 1, orog_path = args.orog_path)
    idx_train, idx_val, idx_test = chronological_splits(X.shape[0])
    np.savez(MODELS_DIR / "splits.npz", train = idx_train, val = idx_val, test = idx_test)

    stats = normalization_stats(X[idx_train], Y[idx_train])
    stats.update({"lr_variables": args.lr_variables, "hr_target": args.hr_target,
                  "orog_path": args.orog_path, "scale_factor": args.scale_factor})
    json.dump(stats, open(MODELS_DIR / "normalization_stats.json", "w"), indent = 2)
    X, Y = normalize(X, Y, stats)

    def loader(idx, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(Y[idx])),
                          batch_size = args.batch_size, shuffle = shuffle)
    train_loader, val_loader = loader(idx_train, True), loader(idx_val, False)

    model = DownscalingModel(in_channels = X.shape[1], lr_shape = X.shape[2:], hr_shape = Y.shape[2:],
                             scale_factor = args.scale_factor,
                             input_number_kernels = args.base_channels,
                             dropout_rate_2D = args.dropout).to(device)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train the UNET to downscale to 3 km resolution")
    parser.add_argument("--lr-pattern", default = LR_PATTERN, help = "glob of (upscaled) predictor files, '{var}' = variable name")
    parser.add_argument("--lr-variables", nargs = "+", default = LR_VARIABLES)
    parser.add_argument("--hr-pattern", default = HR_PATTERN, help = "glob of 3 km target files, '{var}' = variable name")
    parser.add_argument("--hr-target", default = HR_TARGET)
    parser.add_argument("--orog-path", default = OROG_PATH, help = "optional orography file, added as an input channel")
    parser.add_argument("--scale-factor", type = int, default = SCALE_FACTOR, choices = [1, 2, 4, 8],
                        help = "1 = predictors already on the 3 km grid, 4 = raw 12 km predictors")
    parser.add_argument("--base-channels", type = int, default = 32)
    parser.add_argument("--dropout", type = float, default = 0.)
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--epochs", type = int, default = 75)
    parser.add_argument("--patience", type = int, default = 15)
    train(parser.parse_args())

