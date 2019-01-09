import mxnet as mx
from mxnet import sym

def _conv3x3(data, channels, stride):
    return sym.Convolution(data=data, num_filter=channels, kernel=(3, 3), stride=stride, pad=(1, 1))


def residual_block(data, channels, stride, downsample=False):
    bn0 = sym.BatchNorm(data)
    conv1 = _conv3x3(bn0, channels, stride)
    bn1 = sym.BatchNorm(conv1)
    act1 = sym.Activation(bn1, act_type='relu')
    conv2 = _conv3x3(act1, channels, stride)
    bn2 = sym.BatchNorm(conv2)

    if downsample:
        data = sym.Convolution(data=data, kernel=(1,1), stride=stride, no_bias=True, num_filter=channels)

    return sym.Activation(data + bn2, act_type='relu')


def bottle_neck(data, channels, stride, downsample=False):
    bn0 = sym.BatchNorm(data)
    conv1 = sym.Convolution(data, kernel=(
        1, 1), num_filter=channels//4, stride=stride)
    bn1 = sym.BatchNorm(conv1)
    act1 = sym.Activation(bn1, act_type='relu')

    conv2 = _conv3x3(act1, channels=channels//4, stride=stride)
    bn2 = sym.BatchNorm(conv2)
    act2 = sym.Activation(bn2)

    conv3 = sym.Convolution(act2, kernel=(
        1, 1), num_filter=channels, stride=stride)
    bn3 = sym.BatchNorm(conv3)

    if downsample:
        data = sym.Convolution(data=data, kernel=(1,1), stride=stride, no_bias=True, num_filter=channels)

    return sym.Activation(data + bn3, act_type='relu')


class Resnet:
    def __init__(self, block_func, layers, channels, classes=1000):
        assert len(layers) == len(channels) - 1
        assert hasattr(block_func, '__call__')
        input = sym.var('data')
        label = sym.var('softmax_label')
        bn0 = sym.BatchNorm(data=input)
        block0_conv1 = sym.Convolution(data=bn0, num_filter=channels[0], kernel=(7,7), stride=(2,2), pad=(3,3), no_bias=True, name='block0_conv1')
        block0_bn1 = sym.BatchNorm(block0_conv1, name='block0_bn1')
        block0_act1 = sym.Activation(block0_bn1, act_type='relu')
        block0_map = sym.Pooling(block0_act1, pool_type='max', global_pool=True)

        data = block0_map
        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            data = self._make_layer(data, block_func, num_layer, channels[i+1], channels[i], (stride,stride), i+1)
        self.features = data
        gap = sym.Pooling(self.features, pool_type='avg', global_pool=True)
        fc = sym.FullyConnected(gap, num_hidden=classes)
        self.output = sym.SoftmaxOutput(data=fc, label=label, name='softmax_ouput')
    
    def _make_layer(self, data, block_func, num_layer, channels, in_channels, stride, stage_idx):
        data = block_func(data, channels, stride, channels != in_channels)
        for _ in range(num_layer-1):
            data = block_func(data, channels, (1,1))
        return data
    

resnet_spec = {
    18: (residual_block, [2, 2, 2, 2], [64, 64, 128, 256, 512]),
    34: (residual_block, [3, 4, 6, 3], [64, 64, 128, 256, 512]),
    50: (bottle_neck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
    101: (bottle_neck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
    152: (bottle_neck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048])
}
   
def get_resnet(num_layers, **kwargs):
    assert num_layers in resnet_spec, 'Invalid number of layers'
    block_type, layers, channels = resnet_spec[num_layers]
    return Resnet(block_type, layers, channels, **kwargs)


def resnet18(**kwargs):
    return get_resnet(18, **kwargs)

def resnet34(**kwargs):
    return get_resnet(34, **kwargs)

def resnet50(**kwargs):
    return get_resnet(50, **kwargs)

def resnet101(**kwargs):
    return get_resnet(101, **kwargs)

def resnet152(**kwargs):
    return get_resnet(152, **kwargs)