import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from model.deeplabv3 import AtrousPose
from training.datasets.coco import get_loader

LEARNING_RATE = 0.0001
USE_PRETRAINED = False
BTACH_SIZE = 10
RELOAD = False
iter = 0
MAX_ITERS = 9999
ITERS = 0
DISPLAY = 10
config_path = './cfg/yolov3.cfg'
DARKNET_WEIGHTS_FILENAME = './pretrained/darknet53.conv.74'
FREEZE_BACKBONE = False

                #####          ######
                # your dataset path.#
                #####          ######
# data_dir = '/media/hane/DataBase/DL-DATASET/MPII/mpii/mpii_human_pose_v1/images'
# mask_dir = '/media/hane/DataBase/DL-DATASET/MPII/mpii/masks_for_mpii_pose'
# json_path = '/media/hane/DataBase/DL-DATASET/MPII/mpii/MPI.json'
data_dir = 'H:/DL-DATASET/MPII/mpii/mpii_human_pose_v1/images'
mask_dir = 'H:/DL-DATASET/MPII/mpii/masks_for_mpii_pose'
json_path = 'H:/DL-DATASET/MPII/mpii/MPI.json'
# data_dir = '/media/hane/软件/00-Data/rtpose_datasets/COCO/images'
# mask_dir = '/media/hane/软件/01-ComputerVisionEntries/00-real_time_pose_estimation/ZheCao/Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/'
# json_path = '/media/hane/软件/01-ComputerVisionEntries/00-real_time_pose_estimation/ZheCao/Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/json/COCO.json'
                #####          ######
                # your dataset path.#
                #####          ######
                
# parameters of data augmentation
params_transform = {'mode':5, 'scale_min':0.7, 'scale_max':1.6, 'scale_prob':1,
                    'target_dist':1, 'max_rotate_degree':45, 'center_perterb_max':40,
                    'flip_prob':0.5, 'np':56, 'sigma':7.0, 'limb_width':1.0 }


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))

def construct_model(file_name):

    model = AtrousPose()
    if RELOAD:
        model.load_state_dict(torch.load(file_name))
    model_dict = list(model.state_dict())

    if FREEZE_BACKBONE:
        cnt=0
        for i, (name, p) in enumerate(model.named_parameters()):
            if i <= 110: # 137-65 104-50
                p.requires_grad = False

    model_info(model)

    return model.cuda()

# load dataset for networks
def get_dataloader(train=True):
    dataloader = get_loader(json_path, data_dir, # 368 input size, 8 feat_stride
                            mask_dir, 384, 8,  # params_transform is a dictionary
                            'atrous_pose', BTACH_SIZE, params_transform = params_transform,
                            shuffle=True, training=train, num_workers=1, coco=False)

    print('train dataset len1: {}'.format(len(dataloader)))

    return dataloader, len(dataloader)

def get_parameters(model, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]

    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []

    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if 'resnet' not in key:
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': LEARNING_RATE},
              {'params': lr_2, 'lr': LEARNING_RATE * 2.},
              {'params': lr_4, 'lr': LEARNING_RATE * 4.},
              {'params': lr_8, 'lr': LEARNING_RATE * 8.},]

    return params, [1., 2., 4., 8.]

def main(model):

    cudnn.benchmark = True

    train_loader, num_samples1 = get_dataloader()
    # val_loader, num_samples2 = get_dataloader(train=False)
    # print("num_samples:{}".format(num_samples1))
    # print("num_samples:{}".format(num_samples2))
    # loss function
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), LEARNING_RATE)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), LEARNING_RATE)
    # optimizer = AdaBound(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, final_lr=0.1)

    params, multiple = get_parameters(model, False)

    ITERS = 0
    while ITERS < MAX_ITERS:
        train(model, optimizer, train_loader, criterion, multiple)
        # val(model, val_loader, criterion)
        ITERS += 1

def val(model, val_loader, criterion):

    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(10)]

    model.eval()
    with torch.no_grad():
        for step, (image, heatmap, mask_heatmap, vecmap, mask_paf, img_name, idx) in enumerate(val_loader):

            data_time.update(time.time() - end)
            image = image.cuda()
            heatmap = heatmap.cuda()
            vecmap = vecmap.cuda()
            mask_heatmap = mask_heatmap.cuda()
            mask_paf = mask_paf.cuda()
            vec1, heat1 = model(image)

            loss1_1 = criterion(vec1 * mask_paf, vecmap * mask_paf)
            loss1_2 = criterion(heat1 * mask_heatmap, heatmap * mask_heatmap)

            loss = loss1_1 + loss1_2
            losses.update(loss.item(), image.size(0))

            for cnt, l in enumerate([loss1_1, loss1_2]):
                losses_list[cnt].update(l.item(), image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            step += 1
            if step % DISPLAY == 0:
                if step % 10 == 0:
                    with open('./log_loss/dilated_pose_val_loss.txt', 'a') as file:
                        file.write("%.7f" % (losses_list[0].avg) + ' ' + "%.7f" % (losses_list[1].avg) + '\n')
                print('val Iteration: {0}\t'
                      'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                      'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                      'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    step, DISPLAY, batch_time=batch_time,
                    data_time=data_time, loss=losses))
                for cnt in range(0, 2, 2):
                    print('Loss{0}_1 = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                          'Loss{1}_2 = {loss2.val:.8f} (ave = {loss2.avg:.8f})'.format(cnt / 2 + 1, cnt / 2 + 1,
                                                                                       loss1=losses_list[cnt],
                                                                                       loss2=losses_list[cnt + 1]))
                print(time.strftime(
                    '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',
                    time.localtime()))

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(2):
                    losses_list[cnt].reset()

def train(model, optimizer, train_loader, criterion, multiple):
    #global iter
    end = time.time()
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(10)]

    for step, (image, heatmap, mask_heatmap, vecmap, mask_paf, img_name, idx) in enumerate(train_loader):
        # lr = lr * (1 - iter / 200000)
        # print("learning rate:{}".format(lr))
        # for g in optimizer.param_groups:
        #     g['lr'] = lr
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = LEARNING_RATE * multiple[i]

        data_time.update(time.time() - end)
        image = image.cuda()
        heatmap = heatmap.cuda()
        vecmap = vecmap.cuda()
        mask_heatmap = mask_heatmap.cuda()
        mask_paf = mask_paf.cuda()
        vec1, heat1 = model(image)

        loss1_1 = criterion(vec1 * mask_paf, vecmap * mask_paf)
        loss1_2 = criterion(heat1 * mask_heatmap, heatmap * mask_heatmap)

        loss = loss1_1 + loss1_2
        losses.update(loss.item(), image.size(0))

        for cnt, l in enumerate([loss1_1, loss1_2]):
            losses_list[cnt].update(l.item(), image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % 1000 == 0:
            torch.save(model.state_dict(), './legacy/dilated_pose.pth')

        step += 1

        if step % DISPLAY == 0:
            if step % 1000 == 0:
                with open('./log_loss/dilated_pose_train_loss.txt', 'a') as file:
                    file.write("%.7f" % (losses_list[0].avg) + ' ' + "%.7f" % (losses_list[1].avg) + '\n')
            print('Train Iteration: {0}\t'
                  'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                  'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                  'Learning rate = {2}\n'
                  'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                step, DISPLAY, LEARNING_RATE, batch_time=batch_time,
                data_time=data_time, loss=losses))
            for cnt in range(0, 2, 2):
                print('Loss{0}_1 = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                      'Loss{1}_2 = {loss2.val:.8f} (ave = {loss2.avg:.8f})'.format(cnt / 2 + 1, cnt / 2 + 1,
                                                                                   loss1=losses_list[cnt],
                                                                                   loss2=losses_list[cnt + 1]))
            print(time.strftime(
                '%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',
                time.localtime()))

            batch_time.reset()
            data_time.reset()
            losses.reset()
            for cnt in range(2):
                losses_list[cnt].reset()


if __name__ == '__main__':

    model = construct_model('./legacy/dilated_pose.pth')
    torch.cuda.empty_cache()
    main(model)
