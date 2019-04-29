"""MSCOCO Dataloader
   Thanks to @tensorboy @shuangliu
"""

try:
    import ujson as json
except ImportError:
    import json

from torchvision.transforms import ToTensor
# from training.datasets.coco_data.mpii_data_pipeline import Cocokeypoints
from training.datasets.coco_data.mpii import Cocokeypoints
# from training.datasets.coco_data.COCO_data_pipeline import Cocokeypoints
from training.datasets.dataloader import sDataLoader


def get_loader(json_path, data_dir, mask_dir, inp_size, feat_stride, preprocess,
               batch_size, params_transform, training=True, shuffle=True, num_workers=3, coco=True):
    """ Build a COCO dataloader
    :param json_path: string, path to jso file
    :param datadir: string, path to coco data
    :returns : the data_loader
    """
    with open(json_path) as data_file:
        data_this = json.load(data_file)
        data = data_this['root']

    num_samples = len(data)
    train_indexes = []
    val_indexes = []

    for count in range(num_samples):
        if data[count]['isValidation'] != 0.:
            val_indexes.append(count)
        else:
            train_indexes.append(count)

    # fix some format problems
    set1 = set(train_indexes) - set([291, 430, 514, 560, 611, 613, 787, 840, 841, 926, 981, 984, 1167, 1453, 1695, 2092, 2589, 2813, 2826, 3467, 3585, 4274, 4633, 4664, 5042, 5678, 5697, 6084, 6172, 6202, 6459, 7080, 7259, 8003, 8105, 8448, 8458, 8501, 9664, 9776, 11157, 11286, 11682, 12603, 12720, 12727, 13598, 13629, 13631, 13633, 14133, 14309, 15450, 16056, 16062, 16203, 16273, 16315, 16835, 17210, 17302, 17545, 18100, 18402, 18458, 18483, 18918, 19879, 20303, 20500, 20738, 20813, 20837, 21248, 21875, 22130, 22146, 22242, 23114, 23416, 23452, 23640, 23803, 24343, 24424, 24613, 25494, 26421, 26487, 27043, 27086, 27089, 27090, 27092, 27094, 27100, 27184, 27526, 27786, 27819, 28604, 28638, 28721, 28764, 28807])
    set2 = set(val_indexes)- set([904, 4965, 14758, 15187, 15962, 17133, 20015, 20210, 23740, 25349, 27474])
    train_indexes = list(set1)
    val_indexes = list(set2)
    print("train num: ", len(train_indexes))
    print("  val num: ", len(val_indexes))

    coco_data = Cocokeypoints(root=data_dir, mask_dir=mask_dir,
                              index_list=train_indexes if training else val_indexes,
                              data=data, inp_size=inp_size, feat_stride=feat_stride, # feat_stride=8
                              preprocess=preprocess, transform=ToTensor(), params_transform=params_transform)

    data_loader = sDataLoader(coco_data, batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers)

    return data_loader
