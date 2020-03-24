from fastai.vision import *
from src.polygons import pred2poly, lanms, eastify, scale_flow_w_orient
from src.pvanet import conv
from copy import deepcopy


class EastMap(ImagePoints):
    
    def __init__(self, flow:FlowField, scale:bool=True, y_first:bool=False):
        super().__init__(flow=flow, scale=False, y_first=y_first)
        if scale: self._flow = scale_flow_w_orient(self._flow, orient='hw')
        self._data = None

    @classmethod
    def create(cls, polygon:dict, scale:bool=True,
               y_first:bool=False, output_size:int=64) -> 'EastMap':
        """Create EAST instance based on input polygon and optionally output_size"""

        if isinstance(polygon, dict):
            hw = (polygon['height'], polygon['width'])
            ft = torch.tensor([[[polygon['p0x'], polygon['p0y']],
                                [polygon['p1x'], polygon['p1y']],
                                [polygon['p2x'], polygon['p2y']],
                                [polygon['p3x'], polygon['p3y']]]], dtype=torch.float)
        elif isinstance(polygon, (list, np.ndarray)) and all(isinstance(p, dict) for p in polygon):
            hw = polygon[0]['size']
            try:
                ft = torch.stack([p['points'] for p in polygon])
            except TypeError:
                ft = torch.empty(0,4,2)
                scale = False
        else:
            raise RuntimeError("Expected polygon to be a dict or a list of dicts")

        flow = FlowField(hw, ft)
        return cls(flow=flow, y_first=y_first, scale=scale)

    def clone(self):
        "Mimic the behavior of torch.clone for `ImagePoints` objects."
        new_flow = FlowField(size=self.flow.size, flow=self.flow.flow.clone())
        new = EastMap(flow=new_flow, scale=False)
        assert(id(new.flow != id(self.flow)))
        assert(id(new.flow.flow != id(self.flow.flow)))
        return new

    @property
    def data(self)->Tensor:
        "Return the points associated to this object."
        flow = self.flow  # This updates flow before we test if some transforms happened
        if self.transformed:
            # clamping each dim to its size as max, min=0
            assert([*flow.flow.shape][1:] == [4, 2]), f'flow has unexpected shape {flow.flow.shape}'
            flow.flow.clamp_(min=-1, max=1)
            self._data = eastify(flow)
        elif self._data is None:
            self._data = eastify(flow)

        return self._data

    def show(self, y:Image=None, ax:plt.Axes=None, figsize:tuple=(3,3), title:Optional[str]=None, hide_axis:bool=True,
        color:str='red', **kwargs):
        "Show the `EastMap` on `ax`."
        if ax is None: _,ax = plt.subplots(figsize=figsize)
        # polys = self.flow.flow
        
        # check if the flow isn't empty - consisting of all Nones, a dummy rectangle for no detection
        if self.flow.flow.nelement() != 0:
            # TODO: get rid of denormalization
            flow = deepcopy(self.flow)
            flow = scale_flow_w_orient(flow, to_unit=False, orient='hw')
            # flow.add_(1).mul_(tensor(self.flow.size)/2.0)
            # denormalize flow
            # flow = (self.flow.flow+1)*torch.tensor(self.flow.size, dtype=self.flow.flow.dtype)/2
            polys = flow.flow.round().numpy()
            for i, poly in enumerate(polys):
                text=None
                patch = ax.add_patch(patches.Polygon( poly, fill=False, edgecolor=color, lw=2))
                vision.image._draw_outline(patch, 4)
                if text is not None:
                    patch = ax.text(*poly[1], text, rotation=0, verticalalignment='top', color=color, fontsize=14, weight='bold')
                    vision.image._draw_outline(patch,1)


class EastLabelList(ItemList):
    OUTPUT_SIZE = 128
    _processor = PreProcessor
    
    def get(self, i):
        # TODO: move output_size upward to class instantiation
        return EastMap.create(self.items[i], output_size=self.OUTPUT_SIZE)

    def reconstruct(self, t, x):
        if isinstance(t, torch.Tensor):
            proposals = pred2poly(t, min_score=0.99)
        else:
            raise Exception('not implemented')

        merged = lanms(proposals, iou_threshold=0.5, min_votes=1)
        if merged:
            # calculate the scaling factor between detected polygons and original image
            scale_factor = torch.tensor(x.size) / torch.tensor(merged[0].size)
            stacked_flow = torch.stack([p.flow for p in merged])
            stacked_flow = stacked_flow.view(-1, 4, 2).mul(scale_factor)
        else:
            stacked_flow = torch.empty(0,4,2)

        return EastMap(flow=FlowField(size=tuple(x.size[::-1]), flow=stacked_flow),
                       scale=True)


class EastItemList(ImageList):
    _label_cls, _square_show_res = EastLabelList, False
            

class MergingBlock(Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(self, in_c:int, hook:callbacks.hooks.Hook, hook_c:int):
        self.hook = hook
        self.unpool = PixelShuffle_ICNR(in_c)
        
        #out_c is fixed to hook_c 
        out_c = hook_c//2
        self.conv1 = conv(in_c+hook_c, out_c, ks=1)
        self.conv2 = conv(out_c, out_c, ks=3)

    def forward(self, input:Tensor) -> Tensor:
        feature = self.hook.stored
        up_out = self.unpool(input)
        cat_x = torch.cat((up_out, feature), dim=1)
        return self.conv2(self.conv1(cat_x))


class LastLayer(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.conv_score = conv(32, 1, 1, activation=nn.Sigmoid())
        #self.conv_geom = conv(32, 8, 1, activation=nn.Sigmoid())
        self.conv_score = nn.Sequential(nn.Conv2d(32, 1, 1),
                                        nn.Sigmoid())
        self.conv_geom = nn.Sequential(nn.Conv2d(32, 8, 1),
                                       nn.Tanh())
        
    def forward(self, input):
        a = self.conv_score(input)
        b = self.conv_geom(input)
        return torch.cat((a, b), dim=1)


class SimpleLoss(nn.Module):
    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
        clasif_loss = nn.BCELoss(reduction='mean')
        reg_loss = nn.SmoothL1Loss(reduction='mean')

        # regression weighing factor
        n = 1.0
        return clasif_loss(input[:,:1], target[:,:1]) + n*reg_loss(input[:,1:], target[:,1:])


def merge_branch(last_output, hooks, output_size=(64,64)):
    layers = []
    #go through hooks bottom->up
    for hook in hooks[::-1]:
        in_c = last_output[1]
        hook_c = hook.stored.shape[1]
        layer = MergingBlock(in_c, hook, hook_c)

        dummy_size = tuple(last_output[-2:])
        last_output = models.unet.model_sizes(nn.Sequential(layer), size=dummy_size)[0]
        if last_output[-2:]<=output_size:
            layers.append(layer)
    return nn.Sequential(*layers)


def eastlike_net(arch, dummy_size=(256,256), out_ch=32, pretrained=True, cut=None): 
    #TODO dummy size must be here?
    if isinstance(arch, nn.Module):
        encoder = deepcopy(arch)
    else:
        encoder = vision.learner.create_body(arch, pretrained=pretrained, cut=cut)
        
    # run dummy through encoder to get output shapes
    shapes = models.unet.model_sizes(encoder, size=dummy_size)
    feature_map_sizes = [s[-1] for s in shapes]
    out_channels = [s[-3] for s in shapes]

    # get module indices where the feature map size changes
    map_resizing = [n for n, (i, j) in enumerate(zip(feature_map_sizes, feature_map_sizes[1:])) if i!=j]

    # attach hooks to given modules
    hooks = models.unet.hook_outputs([encoder[i] for i in map_resizing])
    # get encoder and hooks shapes
    encoder_out_shape = models.unet.model_sizes(encoder, size=dummy_size)[-1]
    
    return nn.Sequential(encoder,
                         merge_branch(encoder_out_shape, hooks),
                         conv(out_ch, 32, 3),
                         LastLayer()
                        )



