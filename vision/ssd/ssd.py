import mxnet as mx
from mxnet import sym


class SSD:
    def __init__(self, network, base_size, features, channels, sizes, ratios,
                 steps, classes, use_1x1_transition=True, use_bn=True, reduce_ratio=1.0,
                 min_depth=128, global_pool=False, pretrained=False, stds=(0.1, 0.1, 0.2, 0.2),
                 nms_thresh=0.45, nms_topk=400, post_nms=100, anchor_alloc_size=128, **kwargs):
        if network is None:
            num_layers = len(ratios)
        else:
            num_layers = len(features) + len(channels) + int(global_pool)
        
        assert len(sizes) == num_layers + 1
        
        sizes = list(zip(sizes[:-1], sizes[1:]))
        
        assert isinstance(
            ratios, list), 'Must provide ratios as list or list of list'
        
        if not isinstance(ratios[0], (tuple, list)):
            ratios = ratios * num_layers
        
        assert num_layers == len(sizes) == len(ratios),  "Mismatched (number of layers) vs (sizes) vs (ratios): {}, {}, {}".format(
            num_layers, len(sizes), len(ratios))
        
        assert num_layers > 0, "SSD require at least one layer, suggest multiple."

        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        im_size = (base_size, base_size)

        for idx, size, ratio, step in zip(range(num_layers), sizes, ratios, steps):
            anchor_generator = 
