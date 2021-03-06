from torch import nn

from base_network import BaseNetwork


# ============================================
#                  QNetwork
# ============================================
class QNetwork(BaseNetwork):
    """
    In pytorch, unlike tf, it seems like the input shape does
    not need to be specified beforehand. The only thing that
    the Conv2d layer needs in the number of input channels
    
    In tf, for the first layer, we have:
    convLayer1 = tf.keras.layers.Conv2D(
        input_shape,
        data_format,
        filters,
        kernel_size,
        strides,
        activation,
        name
    )

    input_shape = (batchSize, nChannels, nRows, nCols) OR (bs, nR, nC, nChannels)
        depending on what data_format is ('channels_first' or 'channels_last')
    filters = the number of output filters

    In pytorch:
    convLayer1 = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode
    )

    where out_channels -> tf's filters argument. Everything else is the same.

    torch's nn.Sequential is a way of grouping things together. For example,
    torch doesn't automatically apply an activation function like tf does,
    so we have to specify that separately. We can group it with the layer
    so that every time we call the layer, everything contained in the Sequential
    occurs.

    kernel_size and strides can both be either int or tuple. If the value is
    the same in each dimension (e.g., kernel_size=[8,8]) we can just set it
    to 8. Same for strides.

    The data format in pytorch appears to always be (N, C, H, W)

    For a convolutional layer, given an input with shape: (N, C_in, H_in, W_in),
    the output shape is (N, C_out, H_out, W_out) where

    H_out = {[H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1] / stride[0]} + 1

    W_out = {[W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1] / stride[1]} + 1

    For a flatten operation, given a tensor of shape (N, *dims), the output is
    (N, prod(dims)). That is, given (32, 3,4,5,6), the output would be: (32, 360)

    Here, the input images will be (N, 4, 110, 84)
    """
    # -----
    # constructor
    # -----
    def __init__(self, nActions):
        super().__init__()
        # First convolutional layer
        conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=8,
            stride=4
        )
        conv1OutShape = self.get_conv_out_shape(110, 84, conv1)
        # Second convolutional layer
        conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2
        )
        conv2OutShape = self.get_conv_out_shape(conv1OutShape[1], conv1OutShape[2], conv2)
        # First fully connected layer (torch's Linear -> tf's Dense)
        fc1 = nn.Linear(in_features=conv2OutShape.prod(), out_features=256)
        # Output layer
        outputLayer = nn.Linear(in_features=256, out_features=nActions)
        # Network. Note that the last linear activation function isn't
        # needed here as it was in tf. In tf we had to specify an
        # activation, and linear is just g(x) = x
        self.net = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            nn.Flatten(),
            fc1,
            nn.ReLU(),
            outputLayer
        )


    # -----
    # forward
    # -----
    def forward(self, x):
        return self.net(x)
