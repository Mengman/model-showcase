from mxnet import sym


def _conv3x3(data, channles, stride):
  return sym.Convolution(data=data, num_filter=channles, kernel=(3, 3), stride=stride, pad=(1, 1))


def residual_block_v1(data, channels, stride):
  """

  :param data:
  :param channels:
  :param stride:
  :return:
  """
  conv_1 = _conv3x3(data, channels, stride)
  bn_1 = sym.BatchNorm(conv_1)
  act1 = sym.Activation(bn_1, act_type='relu')
  conv_2 = _conv3x3(act1, channels, 1)
  return sym.BatchNorm(conv_2)

def bottle_neck(data, channels, stride):
  conv_1 = sym.Convolution(data=data, kernel=(1,1), num_filter=channels // 4, stride=stride)
  bn_1 = sym.BatchNorm(conv_1)
  act1 = sym.Activation(bn_1, act_type='relu')

  conv_2 = _conv3x3(act1, channles=channels // 4, stride=stride)

