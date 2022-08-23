from __future__ import absolute_import
import resource
from data.util import read_image
import time
from utils.eval_tool import eval_detection_voc
from utils.vis_tool import visdom_bbox, vis_image, vis_bbox
from utils import array_tool as at
from trainer import FasterRCNNTrainer
from torch.utils import data as data_
from model import FasterRCNNVGG16
from data.dataset import Dataset, TestDataset, inverse_normalize, preprocess, pytorch_normalze
from utils.config import opt
import torch as t
from tqdm import tqdm
import matplotlib.pyplot as plt
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
matplotlib.use('agg')
# run this
# python3 -m visdom.server

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit[1])
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# matplotlib.use('agg') #no GUI
matplotlib.use('TkAgg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_,
             gt_difficults_) in tqdm(enumerate(dataloader)):
        print('ii = ', ii, '  ---------------------------')
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [
                                                                       sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


# %%
opt._parse({})

dataset = Dataset(opt)
print('load data')
dataloader = data_.DataLoader(dataset,
                              batch_size=1,
                              shuffle=True, \
                              # pin_memory=True,
                              num_workers=opt.num_workers)
testset = TestDataset(opt)
test_dataloader = data_.DataLoader(testset,
                                   batch_size=1,
                                   num_workers=opt.test_num_workers,
                                   shuffle=True,
                                   pin_memory=True
                                   )


ori_img, bbox, label, difficult = dataset.db.get_example(1)
imgt, bboxt, labelt, scalet = dataset.tsf((ori_img, bbox, label))
# bbox (y_{min}, x_{min}, y_{max}, x_{max})
vis_bbox(ori_img, bbox, label=label, score=None, ax=None)
# vis_image(prepori_img)
plt.show()
# %%

imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_ = testset[1]

# %% testing with trained model
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load(
    'fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth.701052458187')

ori_img2 = read_image('misc/demo.jpg')
print(ori_img2.shape)
ori_img2 = t.from_numpy(ori_img2)[None]
ori_img2.shape


_bboxes, _labels, _scores = trainer.faster_rcnn.predict(
    ori_img2, visualize=True)
vis_bbox(at.tonumpy(ori_img2[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))
# eval_result = eval(test_dataloader, trainer.faster_rcnn, test_num=opt.test_num)

# %% ---------------------------------------------------------------------

faster_rcnn = FasterRCNNVGG16()
print('model construct completed')
time.sleep(2)
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
print('model cuda trainer')
if opt.load_path:
    trainer.load(opt.load_path)
    print('load pretrained model from %s' % opt.load_path)
print('before train viz')
trainer.vis.text(dataset.db.label_names, win='labels')
print('after train viz')
best_map = 0
lr_ = opt.lr

# %%
print('before epoch')
for epoch in range(opt.epoch):
    trainer.reset_meters()
    for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
        print('ii = ', ii)
        scale = at.scalar(scale)
        img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
        trainer.train_step(img, bbox, label, scale)

        if (ii + 1) % opt.plot_every == 0:
            if os.path.exists(opt.debug_file):
                ipdb.set_trace()

            # plot loss
            trainer.vis.plot_many(trainer.get_meter_data())

            # plot groud truth bboxes
            ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            gt_img = visdom_bbox(ori_img_,
                                 at.tonumpy(bbox_[0]),
                                 at.tonumpy(label_[0]))
            trainer.vis.img('gt_img', gt_img)

            # plot predicti bboxes
            _bboxes, _labels, _scores = trainer.faster_rcnn.predict(
                [ori_img_], visualize=True)
            pred_img = visdom_bbox(ori_img_,
                                   at.tonumpy(_bboxes[0]),
                                   at.tonumpy(_labels[0]).reshape(-1),
                                   at.tonumpy(_scores[0]))
            trainer.vis.img('pred_img', pred_img)

            # rpn confusion matrix(meter)
            trainer.vis.text(
                str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
            # roi confusion matrix
            trainer.vis.img(
                'roi_cm',
                at.totensor(
                    trainer.roi_cm.conf,
                    False).float())
    eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
    trainer.vis.plot('test_map', eval_result['map'])
    lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
    log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                              str(eval_result['map']),
                                              str(trainer.get_meter_data()))
    trainer.vis.log(log_info)

    if eval_result['map'] > best_map:
        best_map = eval_result['map']
        best_path = trainer.save(best_map=best_map)
    if epoch == 9:
        trainer.load(best_path)
        trainer.faster_rcnn.scale_lr(opt.lr_decay)
        lr_ = lr_ * opt.lr_decay

    if epoch == 13:
        break