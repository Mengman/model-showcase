import mxnet as mx
from mxnet import sym
from resnet import Resnet

CHANNELS = 256

def _build_pyramid_layer(conv_feat, layer_idx, up_layer=None):
    lateral = sym.Convolution(conv_feat, kernel=(1,1), num_filter=CHANNELS, name='P%d_lateral'%layer_idx)
    if up_layer is None:
        return lateral
    upsample = sym.UpSampling(up_layer, scale=2, num_filter=CHANNELS, sample_type='nearest', num_args=1, name='P%d_upsampling'%(layer_idx+1))
    clip = sym.slice_like(upsample, lateral, 'P%d_clip'%layer_idx)
    p = sym.ElementWiseSum(*[clip, lateral], name='P%d_sum'%layer_idx)
    return sym.Convolution(p, kernel=(3,3), pad=(1,1), num_filter=CHANNELS, name='P%d_aggregate'%layer_idx)


def get_resnet_fpn(conv_feats, subsampling=False):
    num_layer = len(conv_feats) + 1
    pyramids = []
    for i, conv_feat in enumerate(conv_feats):
        up_layer = conv_feats[i-1] if i > 0 else None
        pyramids.append(_build_pyramid_layer(conv_feat, num_layer - i, up_layer))

    if subsampling:
        top = sym.Pooling(pyramids[0], kernel=(3, 3), stride=(2, 2), pad=(1,1), pool_type='max', name='P%d_subsampling'%(num_layer+1))
        pyramids.insert(0, top)
    return pyramids

