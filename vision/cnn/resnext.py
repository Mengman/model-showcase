import mxnet as mx
from mxnet import sym
import math


def _conv3x3(data, channels, stride, group, name=''):
    return sym.Convolution(data, kernel=(3, 3), no_bias=True, stride=(stride, stride),
                           pad=(1, 1), num_filter=channels, num_group=group, name=name)


def block(data, channels, cardinality, bottleneck_width, stride, downsample=False, prefix=''):
    D = int(math.floor(channels * (bottleneck_width / 64)))
    group_width = cardinality * D

    conv1 = sym.Convolution(data, num_filter=group_width,
                            kernel=(1, 1), no_bias=True, name=prefix + '_conv1')
    bn1 = sym.BatchNorm(conv1, name=prefix + '_bn1')
    act1 = sym.Activation(bn1, act_type='relu', name=prefix+'_relu1')
    conv2 = _conv3x3(act1, group_width, stride,
                     cardinality, name=prefix + '_conv2')
    bn2 = sym.BatchNorm(conv2, name=prefix + '_bn2')
    act2 = sym.Activation(bn2, act_type='relu', name=prefix + '_relu2')
    conv3 = sym.Convolution(act2, kernel=(
        1, 1), num_filter=channels * 4, no_bias=True, name=prefix + '_conv3')
    bn3 = sym.BatchNorm(conv3, name=prefix + '_bn3')

    if downsample:
        data = sym.Convolution(data, kernel=(
            1, 1), num_filter=channels * 4, stride=(stride, stride), no_bias=True)
        data = sym.BatchNorm(data)

    return sym.Activation(bn3 + data, act_type='relu', name=prefix + '_relu3')


class ResNext:
    def __init__(self, layers, cardinality, bottleneck_width, classes=1000, **kwargs):
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        channels = 64

        input = sym.var('data')
        label = sym.var('softmax_label')

        stage0_conv1 = sym.Convolution(input, kernel=(7, 7), stride=(
            2, 2), pad=(3, 3), num_filter=channels, no_bias=True, name='stage0_conv1')
        stage0_bn1 = sym.BatchNorm(stage0_conv1, name='stage0_bn1')
        stage0_act1 = sym.Activation(
            stage0_bn1, act_type='relu', name='stage0_relu1')
        stage0_map = sym.Pooling(stage0_act1, pool_type='max', kernel=(
            3, 3), stride=(2, 2), pad=(1, 1), name='stage0_map')

        data = stage0_map
        self.features = []
        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            data = self.__make_layer(data, channels, num_layer, stride, i + 1)

            channels *= 2
            self.features.append(data)

        gap = sym.Pooling(self.features, pool_type='avg', global_pool=True)
        fc = sym.FullyConnected(gap, num_hidden=classes)
        self.output = sym.SoftmaxOutput(fc, label=label, name='softmax_output')

    def __make_layer(self, data, channels, num_layer, stride, stage_index):
        prefix = 'stage{}'.format(stage_index)
        data = block(data, channels, self.cardinality, self.bottleneck_width,
                     stride, downsample=True, prefix=prefix + '_block1')

        for i in range(num_layer-1):
            data = block(data, channels, self.cardinality, self.bottleneck_width,
                         1, False, prefix + '_block{}'.format(i+2))
        return data


resnext_spec = {50: [3, 4, 6, 3],
                101: [3, 4, 23, 3]}


def get_resnext(num_layers, cardinality=32, bottleneck_width=4, **kwargs):
    assert num_layers in resnext_spec
    resnext = ResNext(resnext_spec[num_layers],
                      cardinality, bottleneck_width, **kwargs)
    return resnext


def resnext50_32x4d(**kwargs):
    return get_resnext(50)


def resnext101_32x4d(**kwargs):
    return get_resnext(101)
