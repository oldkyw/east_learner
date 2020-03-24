from fastai.vision import *
from src.east import eastlike_net, EastItemList, SimpleLoss
from src.metric import TestScoreMetric
from src.pvanet import pvanet
from src.polygons import parse_json


img_path = Path('./dataset/train')
test_path = Path('./dataset/test/img')

# Dataset init
parse_ds = lambda path: parse_json(path.parent/(path.stem+'.json'), array=torch.tensor)
tfms = get_transforms(do_flip=False, max_rotate=5, max_zoom=1.2, max_lighting=0.3, max_warp=0.2)
data = (EastItemList.from_folder(img_path)
        .split_by_rand_pct(0.1)
        .label_from_func(parse_ds)
        .transform(tfms=tfms, tfm_y=True, size=512)
        .databunch(bs=4))

# Model specification
cut_func = lambda m: list(m.children())[0]
model = eastlike_net(pvanet(), out_ch=32, cut=cut_func)
test_wrapper = lambda learn: TestScoreMetric(learn, test_path=test_path)

# Learner definition
learn = Learner(data, model, loss_func=SimpleLoss(), callback_fns=test_wrapper)

# Training begin
learn.unfreeze()
learn.fit_one_cycle(100, 0.001)

