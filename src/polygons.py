from pathlib import Path
import json
import copy
import warnings

import torch
from fastai.vision import Image, FlowField
import cv2
import numpy as np


class Polygon(FlowField):

    def __init__(self, flow:torch.Tensor, size:tuple, score:float=1.0, votes:int=1, _map=None):
        assert(len(size)==2)
        super().__init__(flow=flow, size=size)
        self.score = score
        self.votes = votes
        self._map = _map

    def refresh(self):
        self._map = eastify(flow=FlowField(size=self.size, flow=self.flow), just_mask=True, denorm=False)

    @property
    def map(self):
        if self._map is None:
            self.refresh()
        return self._map

    def merge_(self, b):
        assert (self.size == b.size), 'Cannot merge cause of inconsistent sizes'
        score_ratio = b.score / self.score
        self.score += b.score
        self.flow.mul_(1 / (1 + score_ratio)).add_(b.flow.mul(1 / (1 + 1 / score_ratio)))
        self.votes += b.votes
        self.refresh()
        return self

    def merge(self, b):
        assert (self.size == b.size), 'Cannot merge cause of inconsistent sizes'
        score_ratio = b.score / self.score
        score = self.score + b.score
        flow = self.flow.mul(1 / (1 + score_ratio)).add(b.flow.mul(1 / (1 + 1 / score_ratio)))
        votes = self.votes + b.votes
        return Polygon(flow=flow, score=score, size=self.size, votes=votes)

    def show(self, size=256):
        return Image(self.map.view(self.size).unsqueeze(0)).resize(size)

    def clone(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f'({self.score:.3}, {self.votes}) {self.flow}'

    def __eq__(self, other):
        return torch.equal(self.flow, other.flow) & (self.size == other.size) & \
               (self.score == other.score) & (self.votes == other.votes)

    def __hash__(self):
        return hash((self.flow, self.size, self.score, self.votes))


class Target(Polygon):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = {}

    @classmethod
    def from_polygon(cls, poly: Polygon):
        return cls(**poly.__dict__)

    def add_detections(self, predictions: [Polygon], iou_threshold=0.5):
        for polygon in predictions:
            overlap = iou_single_vs_single(self, polygon)
            if overlap > iou_threshold:
                self._detections[polygon] = overlap
        return self

    @property
    def detections(self):
        return self._detections


def iou_single_vs_single(a: Polygon, b: Polygon):
    inter = len((a.map * b.map).nonzero())
    union = len((a.map + b.map).nonzero())
    # epsilon added for numerical stability
    try:
        return inter/union
    except ZeroDivisionError:
        return 0


def iou_single_vs_many(a: Polygon, b: [Polygon]):
    if not b: return torch.empty(0)
    bm = torch.stack([p.map for p in b])
    am = a.map.repeat(len(b), 1)

    inter = ((am*bm) != 0).sum(dim=1, dtype=torch.float)
    union = ((am+bm) != 0).sum(dim=1, dtype=torch.float)
    res = inter/union
    return res


def lanms(proposals: [Polygon], iou_threshold=0.5, min_votes=4, min_votes_pct=None):
    """Function that applies Locally Aware NMS and merges polygons given in proposals if their IoU
    is greater than iou_threshold. The output is a list of merged polygons
    """

    if min_votes_pct is not None:
        if proposals:
            n_pixels = proposals[0].size[0] * proposals[0].size[1]
            min_votes = min_votes_pct * n_pixels

    left = proposals
    while True:
        merged = False
        right = []
        for this in left:
            res = iou_single_vs_many(this, right)
            res *= res >= iou_threshold

            # check if there is at least one polygon meeting the criteria
            if res.nonzero().nelement():
                # TODO: try using len(left) == len(right) instead of merged:bool
                match = res.max(dim=0)[1]
                right[match].merge_(this)
                merged = True
            else:
                right.append(this.clone())

        # replace working list with the one already merged
        left = right
        if not merged: break
    return [polygon for polygon in left if polygon.votes>=min_votes]


def pred2poly(pred:torch.Tensor, min_score:float=0.8):
    assert(pred.ndim != 4), 'probably using score_threshold on batch'
    # [9, 985, 1200]
    # [n_c, h, w]
    n_c, h, w = pred.shape
    t = pred.clone()

    lr = torch.linspace(0, w - 1, w, device=t.device)
    tb = torch.linspace(0, h - 1, h, device=t.device).view(-1, 1)
    # skip score mask & select just corners geometry from the prediction tensor 1:8
    geom = t.narrow(-3, 1, n_c - 1)

    # run operations on geometry to recover location of corners
    # add decremental linspace left right for x_dist channels
    geom[::2].mul_(-w).add_(lr)
    # add decremental linspace top bottom for y_dist channels
    geom[1::2].mul_(-h).add_(tb)

    # transpose and reshape from B, C, H, W to B, H*W, C
    t = t.transpose(-2, -1).transpose(-3, -1)
    t = t.view(h * w, n_c)
    # t = t.view(-1, size * size, n_c); filtered = [ti[ti[:, 0] > min_score] for ti in t]

    filtered = t[t[:, 0] > min_score]
    polygons = [Polygon(flow=p[1:], size=(h, w), score=p[0]) for p in filtered]
    # polygons = [[Polygon(flow=p[1:], size=(size, size), score=p[0]) for p in bn] for bn in filtered]
    return polygons


def _dist_y(size, indice, dtype=torch.float):
    h, w = size
    return torch.linspace(start=(-indice), end=(h - 1 - indice),
                          steps=h, dtype=dtype).view(-1, 1).repeat(1, w)


def _dist_x(size, indice, dtype=torch.float):
    h, w = size
    return torch.linspace(start=(-indice), end=(w - 1 - indice), steps=w,
                          dtype=dtype).repeat(h, 1)


def eastify(flow:FlowField, dtype=torch.float, just_mask: bool = False,
            denorm=True) -> torch.Tensor:
    "Create a stacked 9 channel EAST-type output mask based on size and vertices"
    ntype = {torch.float32: 'float32',
             torch.float64: 'float64',
             int: 'int32'}

    hw = flow.size
    local_flow = copy.deepcopy(flow)
    # denormalize flow
    if denorm:
        local_flow = scale_flow_w_orient(local_flow, to_unit=False)
    polygons = local_flow.flow

    # scaling down so the output map is not as large as the image
    scale_factor = 4.0
    # hw = torch.tensor(hw[::-1]).div(scale_factor).int().tolist()
    hw = torch.tensor(hw).div(scale_factor).int().tolist()
    polygons.div_(scale_factor)

    # round values, assure shape, convert to numpy with correct type
    polygons = polygons.round().cpu().view(-1, 4, 2).numpy().astype(ntype[int])

    # Prepare canvas
    mask_aggregate = torch.zeros(size=hw)
    geometry = torch.zeros_like(mask_aggregate).repeat(8, 1, 1)

    for n_poly, polygon in enumerate(polygons):
        single_mask = np.zeros(shape=hw, dtype=ntype[dtype])
        cv2.fillConvexPoly(single_mask, polygon, color=1)
        single_mask = torch.from_numpy(single_mask)

        if not just_mask:
            for n_vertice, (x, y) in enumerate(polygon):
                geometry[n_vertice*2].add_(_dist_x(hw, x, dtype=dtype).mul(single_mask))
                geometry[n_vertice*2 + 1].add_(_dist_y(hw, y, dtype=dtype).mul(single_mask))
        mask_aggregate.add_(single_mask)

    # Stop if expected output is mask
    if just_mask: return mask_aggregate.view(-1)

    # Normalize geometry
    norms = torch.tensor(hw[::-1]).repeat(4).view(-1,1,1)
    geometry.div_(norms)

    # Final assertions
    assert(geometry.max() <= 1 and geometry.min() >= -1)
    mask_aggregate.clamp_(0.0, 1.0)
    assert(mask_aggregate.max() <= 1), f'{mask_aggregate[mask_aggregate>1]}'

    return torch.cat([mask_aggregate.unsqueeze(0), geometry])


def scale_flow_w_orient(flow, to_unit=True, orient='hw', y_first=False) -> FlowField:
    "Scale the coords in `flow` to -1/1 or the image size depending on `to_unit`."
    if orient == 'hw':
        h, w = flow.size
    elif orient == 'wh':
        w, h = flow.size
    else:
        raise RuntimeError(f'Unknown orient specified: {orient}')

    s = torch.tensor([w/2, h/2])[None]
    if y_first: s = s.flip(-1)

    if to_unit:
        flow.flow = flow.flow/s-1
    else:
        flow.flow = (flow.flow+1)*s
    return flow


def parse_json(path, array=np.array):
    path = Path(path)
    if path.suffix == '.jpg':
        json_path = path.parent.parent / f'json/{path.stem}.json'
    elif path.suffix == '.json':
        json_path = path

    with open(json_path, encoding='latin1') as json_file:
        data = json.load(json_file)

    h = data['imageHeight']
    w = data["imageWidth"]

    items = dict()
    for item in data['shapes']:
        assert (len(item['label'].split('__')) == 3), f'label different than expected {json_path}'
        value, label, _ = item['label'].split('__')
        points = array(item['points'])
        assert (points.shape[1] == 2)
        try:
            items[label.lower()].append({'value': value, 'points': points,
                                         'size': (h, w)})
        except KeyError:
            items[label.lower()] = [{'value': value, 'points': points,
                                     'size': (h, w)}]

    blank = [{'value': None, 'points': None, 'size': (h, w)}]
    if 'linenumber' in items:
        for k in items['linenumber']:
            assert ((len(k['points'])) == 4), f'{path}'
    return items['linenumber'] if 'linenumber' in items else blank