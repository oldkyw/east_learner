from fastai.vision import *
from src.polygons import pred2poly, lanms, Target, parse_json
from src.east import EastItemList


class TestScoreMetric(LearnerCallback):
    _order = -22

    def __init__(self, learn, test_path):
        super().__init__(learn)
        self.name = 'test_score'
        self.device = torch.device('cpu')
        self.test_path = test_path

        # Dynamic alignment of test_ds size to the training_ds
        try:
            target_size = self.learn.data.train_ds.tfmargs['size']
        except KeyError:
            raise RuntimeError(f'Cannot resolve target size for test set')
            target_size = learn.data.train_ds.x[0].size[0]

        parse_test = lambda path: parse_json(path, array=torch.tensor)
        self.test_ds = (EastItemList.from_folder(self.test_path)
                        .split_none()
                        .label_from_func(parse_test)
                        .transform(tfms=[None, None], tfm_y=True, size=target_size)
                        .databunch(bs=16)
                        )

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names([self.name])

    def on_epoch_end(self, last_metrics, **kwargs):
        dl = self.test_ds.train_dl
        tp, fn, fp = 0.0, 0.0, 0.0
        with torch.no_grad():
            for xb, target in dl:
                output = self.learn.model(xb)
                for one_output, one_target in zip(output.to(self.device), target.to(self.device)):
                    pred_polys = set(lanms(pred2poly(one_output)))
                    target_polys = (Target.from_polygon(p).add_detections(pred_polys) for p in lanms(pred2poly(one_target)))

                    for t in target_polys:
                        if t.detections:
                            tp += 1
                        else:
                            fn += 1
                        for d in t.detections:
                            pred_polys.discard(d)

                    fp += 1.0*len(pred_polys)

        if torch.distributed.is_initialized(): torch.distributed.barrier()
        # deletion for python gb collector
        del output
        try:
            f1_score = 2.0 * tp / (2.0 * tp + fp + fn)
        except ZeroDivisionError:
            f1_score = -1
        return add_metrics(last_metrics, f1_score)
