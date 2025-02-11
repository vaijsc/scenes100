#!python3

import os
import sys
import types
import time
import datetime
import gc
import json
import copy
import gzip
import math
import random
import tqdm
import glob
import psutil
import argparse

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import scipy.io

import torch
import torch.utils.data as torchdata

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, create_ddp_model
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import logging
import weakref
import platform


thing_classes = ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', '004.Groove_billed_Ani', '005.Crested_Auklet', '006.Least_Auklet', '007.Parakeet_Auklet', '008.Rhinoceros_Auklet', '009.Brewer_Blackbird', '010.Red_winged_Blackbird', '011.Rusty_Blackbird', '012.Yellow_headed_Blackbird', '013.Bobolink', '014.Indigo_Bunting', '015.Lazuli_Bunting', '016.Painted_Bunting', '017.Cardinal', '018.Spotted_Catbird', '019.Gray_Catbird', '020.Yellow_breasted_Chat', '021.Eastern_Towhee', '022.Chuck_will_Widow', '023.Brandt_Cormorant', '024.Red_faced_Cormorant', '025.Pelagic_Cormorant', '026.Bronzed_Cowbird', '027.Shiny_Cowbird', '028.Brown_Creeper', '029.American_Crow', '030.Fish_Crow', '031.Black_billed_Cuckoo', '032.Mangrove_Cuckoo', '033.Yellow_billed_Cuckoo', '034.Gray_crowned_Rosy_Finch', '035.Purple_Finch', '036.Northern_Flicker', '037.Acadian_Flycatcher', '038.Great_Crested_Flycatcher', '039.Least_Flycatcher', '040.Olive_sided_Flycatcher', '041.Scissor_tailed_Flycatcher', '042.Vermilion_Flycatcher', '043.Yellow_bellied_Flycatcher', '044.Frigatebird', '045.Northern_Fulmar', '046.Gadwall', '047.American_Goldfinch', '048.European_Goldfinch', '049.Boat_tailed_Grackle', '050.Eared_Grebe', '051.Horned_Grebe', '052.Pied_billed_Grebe', '053.Western_Grebe', '054.Blue_Grosbeak', '055.Evening_Grosbeak', '056.Pine_Grosbeak', '057.Rose_breasted_Grosbeak', '058.Pigeon_Guillemot', '059.California_Gull', '060.Glaucous_winged_Gull', '061.Heermann_Gull', '062.Herring_Gull', '063.Ivory_Gull', '064.Ring_billed_Gull', '065.Slaty_backed_Gull', '066.Western_Gull', '067.Anna_Hummingbird', '068.Ruby_throated_Hummingbird', '069.Rufous_Hummingbird', '070.Green_Violetear', '071.Long_tailed_Jaeger', '072.Pomarine_Jaeger', '073.Blue_Jay', '074.Florida_Jay', '075.Green_Jay', '076.Dark_eyed_Junco', '077.Tropical_Kingbird', '078.Gray_Kingbird', '079.Belted_Kingfisher', '080.Green_Kingfisher', '081.Pied_Kingfisher', '082.Ringed_Kingfisher', '083.White_breasted_Kingfisher', '084.Red_legged_Kittiwake', '085.Horned_Lark', '086.Pacific_Loon', '087.Mallard', '088.Western_Meadowlark', '089.Hooded_Merganser', '090.Red_breasted_Merganser', '091.Mockingbird', '092.Nighthawk', '093.Clark_Nutcracker', '094.White_breasted_Nuthatch', '095.Baltimore_Oriole', '096.Hooded_Oriole', '097.Orchard_Oriole', '098.Scott_Oriole', '099.Ovenbird', '100.Brown_Pelican', '101.White_Pelican', '102.Western_Wood_Pewee', '103.Sayornis', '104.American_Pipit', '105.Whip_poor_Will', '106.Horned_Puffin', '107.Common_Raven', '108.White_necked_Raven', '109.American_Redstart', '110.Geococcyx', '111.Loggerhead_Shrike', '112.Great_Grey_Shrike', '113.Baird_Sparrow', '114.Black_throated_Sparrow', '115.Brewer_Sparrow', '116.Chipping_Sparrow', '117.Clay_colored_Sparrow', '118.House_Sparrow', '119.Field_Sparrow', '120.Fox_Sparrow', '121.Grasshopper_Sparrow', '122.Harris_Sparrow', '123.Henslow_Sparrow', '124.Le_Conte_Sparrow', '125.Lincoln_Sparrow', '126.Nelson_Sharp_tailed_Sparrow', '127.Savannah_Sparrow', '128.Seaside_Sparrow', '129.Song_Sparrow', '130.Tree_Sparrow', '131.Vesper_Sparrow', '132.White_crowned_Sparrow', '133.White_throated_Sparrow', '134.Cape_Glossy_Starling', '135.Bank_Swallow', '136.Barn_Swallow', '137.Cliff_Swallow', '138.Tree_Swallow', '139.Scarlet_Tanager', '140.Summer_Tanager', '141.Artic_Tern', '142.Black_Tern', '143.Caspian_Tern', '144.Common_Tern', '145.Elegant_Tern', '146.Forsters_Tern', '147.Least_Tern', '148.Green_tailed_Towhee', '149.Brown_Thrasher', '150.Sage_Thrasher', '151.Black_capped_Vireo', '152.Blue_headed_Vireo', '153.Philadelphia_Vireo', '154.Red_eyed_Vireo', '155.Warbling_Vireo', '156.White_eyed_Vireo', '157.Yellow_throated_Vireo', '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler', '160.Black_throated_Blue_Warbler', '161.Blue_winged_Warbler', '162.Canada_Warbler', '163.Cape_May_Warbler', '164.Cerulean_Warbler', '165.Chestnut_sided_Warbler', '166.Golden_winged_Warbler', '167.Hooded_Warbler', '168.Kentucky_Warbler', '169.Magnolia_Warbler', '170.Mourning_Warbler', '171.Myrtle_Warbler', '172.Nashville_Warbler', '173.Orange_crowned_Warbler', '174.Palm_Warbler', '175.Pine_Warbler', '176.Prairie_Warbler', '177.Prothonotary_Warbler', '178.Swainson_Warbler', '179.Tennessee_Warbler', '180.Wilson_Warbler', '181.Worm_eating_Warbler', '182.Yellow_Warbler', '183.Northern_Waterthrush', '184.Louisiana_Waterthrush', '185.Bohemian_Waxwing', '186.Cedar_Waxwing', '187.American_Three_toed_Woodpecker', '188.Pileated_Woodpecker', '189.Red_bellied_Woodpecker', '190.Red_cockaded_Woodpecker', '191.Red_headed_Woodpecker', '192.Downy_Woodpecker', '193.Bewick_Wren', '194.Cactus_Wren', '195.Carolina_Wren', '196.House_Wren', '197.Marsh_Wren', '198.Rock_Wren', '199.Winter_Wren', '200.Common_Yellowthroat']
finetune_output = os.path.join(os.path.dirname(__file__), 'finetune_output_cub200')


def get_cub200_detection(cub200dir):
    def read_filelist(txtfile):
        print('parsing %s ...' % txtfile)
        with open(txtfile, 'r') as fp:
            lines = list(map(lambda s: s.strip().split('/'), fp.readlines()))
        filelist = []
        for f1, f2 in lines:
            lb = thing_classes.index(f1)
            assert lb >= 0 and f2[-4:] == '.jpg'
            image_file = os.path.join(cub200dir, 'images', f1, f2)
            bbox_file = os.path.join(cub200dir, 'annotations-mat', f1, f2[:-4] + '.mat')
            assert os.access(image_file, os.R_OK), image_file
            assert os.access(bbox_file, os.R_OK), bbox_file
            filelist.append({
                'label': lb,
                'image_file': image_file,
                'bbox_file': bbox_file,
            })
        dst_dicts = []
        for f in tqdm.tqdm(filelist, ascii=True):
            bbox = scipy.io.loadmat(f['bbox_file'])['bbox']
            assert bbox.shape == (1, 1)
            x1, y1, x2, y2 = bbox[0, 0]
            assert x1.shape == y1.shape == x2.shape == y2.shape == (1, 1)
            x1, y1, x2, y2 = map(lambda _x: int(_x[0, 0]), [x1, y1, x2, y2])
            im_arr = skimage.io.imread(f['image_file'])
            H, W, C = im_arr.shape
            assert C == 3
            dst_dicts.append({
                'image_id': len(dst_dicts) + 1,
                'file_name': f['image_file'],
                'width': W,
                'height': H,
                'annotations': [{
                    'category_id': f['label'],
                    'bbox': [x1, y1, x2, y2],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [],
                }],
            })
        # import matplotlib.patches as patches
        # for i in range(0, 5):
        #     _d = dst_dicts[random.randint(0, len(dst_dicts))]
        #     _, ax = plt.subplots()
        #     ax.imshow(skimage.io.imread(_d['file_name']))
        #     x1, y1, x2, y2 =_d['annotations'][0]['bbox']
        #     ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none'))
        #     ax.set_title(str(_d))
        #     plt.show()
        return dst_dicts

    uname = platform.uname()
    cache_json = os.path.join(cub200dir, 'detectron2_dataset_cache_%s_%s.json' % (uname.system.lower(), uname.node.lower()))
    if os.access(cache_json, os.R_OK):
        with open(cache_json, 'r') as fp:
            datasets = json.load(fp)
        print('loaded from cached', cache_json)
    else:
        train_dicts = read_filelist(os.path.join(cub200dir, 'lists', 'train.txt'))
        valid_dicts = read_filelist(os.path.join(cub200dir, 'lists', 'test.txt'))
        datasets = {'train': train_dicts, 'valid': valid_dicts}
        with open(cache_json, 'w') as fp:
            json.dump(datasets, fp)
    print('CUB200 detection training split:   %d images' % len(datasets['train']))
    print('CUB200 detection validation split: %d images' % len(datasets['valid']))
    return datasets['train'], datasets['valid']


# DefaultTrainer._trainer is instance of SimpleTrainer
# DefaultTrainer & SimpleTrainer are subclass of TrainerBase
def finetune_simple_trainer_run_step(self):
    assert self.model.training, '[SimpleTrainer] model was changed to eval mode!'
    start = time.perf_counter()
    data = next(self._data_loader_iter)
    data_time = time.perf_counter() - start
    loss_dict = self.model(data)
    loss_dict_items = {k: loss_dict[k].item() for k in loss_dict}
    if isinstance(loss_dict, torch.Tensor):
        losses = loss_dict
        loss_dict = {'total_loss': loss_dict}
    else:
        losses = sum(loss_dict.values())
    self.optimizer.zero_grad()
    losses.backward()
    self._write_metrics(loss_dict, data_time)
    # If you need gradient clipping/scaling or other processing, you can wrap the optimizer with your custom `step()` method. But it is suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
    self.optimizer.step()
    self.loss_history.append({'iter': self.iter, 'loss': loss_dict_items})
    self.lr_history.append({'iter': self.iter, 'lr': float(self.optimizer.param_groups[0]['lr'])})


# wrap detectron2/engine/defaults.py:DefaultTrainer
class FinetuneTrainer(DefaultTrainer):
    def __init__(self, cfg, gmm_models=None):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger('detectron2')
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            detectron2.utils.logger.setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, detectron2.utils.comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = detectron2.checkpoint.DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())
        assert isinstance(self._trainer, SimpleTrainer), 'self._trainer class mismatch'
        self.exception_count, self._trainer.lr_history, self._trainer.loss_history = 0, [], []

    def build_hooks(self):
        ret = super().build_hooks()
        self.eval_results_all = {}
        def test_and_save_results_save():
            self._last_eval_results = self.test(self.cfg, self.model)
            self.eval_results_all[self.iter] = copy.deepcopy(self._last_eval_results)
            return self._last_eval_results
        for i in range(0, len(ret)):
            if isinstance(ret[i], detectron2.engine.hooks.EvalHook):
                ret[i] = detectron2.engine.hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_save)
        return ret

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=finetune_output)


def train(args):
    desc_train, desc_valid = 'cub200_train', 'cub200_valid'
    dst_train, dst_valid = get_cub200_detection(args.cub200dir)
    DatasetCatalog.register(desc_train, lambda: dst_train)
    MetadataCatalog.get(desc_train).thing_classes = thing_classes
    DatasetCatalog.register(desc_valid, lambda: dst_valid)
    MetadataCatalog.get(desc_valid).thing_classes = thing_classes

    if args.model == 'r50-fpn-3x':
        model_yaml = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    if args.model == 'r101-fpn-3x':
        model_yaml = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size
    cfg.SOLVER.IMS_PER_BATCH = args.image_batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WARMUP_ITERS = args.iters // 10
    cfg.SOLVER.GAMMA = 0.5
    cfg.SOLVER.STEPS = (args.iters // 3, args.iters * 2 // 3)
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.OUTPUT_DIR = finetune_output
    cfg.DATASETS.TRAIN = (desc_train,)
    cfg.DATASETS.TEST = (desc_valid,)
    print('- input channel format:', cfg.INPUT.FORMAT)
    print('- load weights from:', cfg.MODEL.WEIGHTS)
    print('- test score threshold:', cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    print('- object classes:', cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    import detectron2.evaluation.evaluator
    detectron2.evaluation.evaluator.evaluate_interval_n = 120
    import detectron2.engine.defaults
    detectron2.engine.defaults.default_trainer_log_period = 120

    trainer = FinetuneTrainer(cfg)
    assert isinstance(trainer._trainer, SimpleTrainer), 'trainer class mismatch'
    trainer._trainer.run_step = types.MethodType(finetune_simple_trainer_run_step, trainer._trainer)
    trainer.resume_or_load(resume=False)

    prefix = 'cub200_detection_%s' % args.model
    for idx, dataset_name in enumerate(trainer.cfg.DATASETS.TEST):
        print('Evaluate on %s' % dataset_name)
        data_loader = trainer.build_test_loader(trainer.cfg, dataset_name)
        evaluator = trainer.build_evaluator(trainer.cfg, dataset_name)
        results_0 = inference_on_dataset(trainer.model, data_loader, evaluator)
    del data_loader, evaluator
    trainer.eval_results_all[0] = results_0
    trainer.train()

    if not detectron2.utils.comm.is_main_process():
        print('in sub-process, exiting')
        return
    with open(os.path.join(os.path.dirname(__file__), prefix + '.json'), 'w') as fp:
        json.dump({'results': trainer.eval_results_all, 'args': vars(args), 'lr_history': trainer._trainer.lr_history, 'loss_history': trainer._trainer.loss_history}, fp)
    m = trainer.model
    if isinstance(m, torch.nn.DataParallel) or isinstance(m, torch.nn.parallel.DistributedDataParallel):
        print('unwrap data parallel')
        m = m.module
    torch.save(m.state_dict(), os.path.join(os.path.dirname(__file__), prefix + '.pth'))

    aps, lr_history, loss_history = trainer.eval_results_all, trainer._trainer.lr_history, trainer._trainer.loss_history
    iter_list = sorted(list(aps.keys()))
    dst_mAP = [aps[i]['bbox']['AP'] for i in iter_list]
    dst_AP50 = [aps[i]['bbox']['AP50'] for i in iter_list]
    dst_mAP = [x if not math.isnan(x) else 0.0 for x in dst_mAP]
    dst_AP50 = [x if not math.isnan(x) else 0.0 for x in dst_AP50]

    lr_history = np.array([[x['iter'], x['lr']] for x in lr_history])
    loss_history_dict, smooth_L = {}, 32
    for loss_key in loss_history[0]['loss']:
        loss_history_dict[loss_key] = np.array([[x['iter'], x['loss'][loss_key]] for x in loss_history])
        for i in range(smooth_L, loss_history_dict[loss_key].shape[0]):
            loss_history_dict[loss_key][i, 1] = loss_history_dict[loss_key][i - smooth_L : i + 1, 1].mean()
        loss_history_dict[loss_key] = loss_history_dict[loss_key][smooth_L + 1 :, :]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(lr_history[:, 0], lr_history[:, 1] / lr_history[:, 1].max(), linestyle='--', color='#000000')
    plt.plot(iter_list, np.array(dst_AP50) / 100, linestyle='-', marker='o', color='#FF0000')
    plt.plot(iter_list, np.array(dst_mAP) / 100, linestyle='-', marker='o', color='#0000FF')
    plt.legend(['lr ($\\times$%.1e)' % lr_history[:, 1].max(), 'Valid AP50', 'Valid mAP'])
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel('Training Iterations')
    plt.title('AP')

    plt.subplot(1, 2, 2)
    colors, color_i = ['#EE0000', '#00EE00', '#0000EE', '#AAAA00', '#00AAAA', '#AA00AA', '#000000'], 0
    legends = []
    for loss_key in loss_history_dict:
        plt.plot(loss_history_dict[loss_key][:, 0], loss_history_dict[loss_key][:, 1], linestyle='-', color=colors[color_i])
        legends.append(loss_key)
        color_i += 1
    plt.legend(legends)
    plt.grid(True)
    plt.xlim(max(iter_list) * -0.02, max(iter_list) * 1.02)
    plt.xlabel('Training Iterations')
    plt.title('losses')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), prefix + '.pdf'))
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--model', type=str, choices=['r50-fpn-3x', 'r101-fpn-3x'])
    parser.add_argument('--cub200dir', type=str)
    parser.add_argument('--iters', type=int)
    parser.add_argument('--eval_interval', type=int)
    parser.add_argument('--image_batch_size', default=8, type=int)
    parser.add_argument('--roi_batch_size', default=64, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()
    args.cub200dir = os.path.normpath(args.cub200dir)
    print(args)

    if not os.access(finetune_output, os.W_OK):
        os.mkdir(finetune_output)
    assert os.path.isdir(finetune_output)
    train(args)


'''
python finetune_cub200.py --model r50-fpn-3x --cub200dir ../../../CUB200 --num_workers 0 --image_batch_size 3 --iters 240 --eval_interval 63
python finetune_cub200.py --model r50-fpn-3x --cub200dir ../../../CUB200 --num_workers 8 --image_batch_size 8 --iters 40000 --eval_interval 2001

python finetune_cub200.py --model r101-fpn-3x --cub200dir ../../../CUB200 --num_workers 0 --image_batch_size 3 --iters 240 --eval_interval 300
python finetune_cub200.py --model r101-fpn-3x --cub200dir ../../../CUB200 --num_workers 8 --image_batch_size 6 --iters 50000 --eval_interval 2501
'''