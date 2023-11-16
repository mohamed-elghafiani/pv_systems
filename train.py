import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from dataset import loader
from model import Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_once_data(num):
    Iter = 1234

    torch.manual_seed(Iter)
    EPOCH = 1000
    LR = 0.01

    model = Model()

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(DEVICE)

    opt = torch.optim.SGD(model.parameters(), lr=LR,
                        momentum=0.9, weight_decay=1e-08)
    # opt = torch.optim.Adam(model.parameters(), lr=LR)

    save_dir = os.path.join(os.getcwd(), f'checkpoint{num}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log = open(os.path.join(save_dir, 'log.txt'), 'a')

    def print_log(print_string, log):
        print("{}".format(print_string))
        log.write('{}\n'.format(print_string))
        log.flush()

    def save_checkpoint(state, is_best, epoch):
        """
        Save the training model
        """
        if is_best:
            max_file_num = 0
            filelist = os.listdir(save_dir)
            filelist.sort()
            file_num = 0
            to_move = []
            for file in filelist:
                if 'model' in file:
                    file_num = file_num + 1
                    to_move.append(os.path.join(save_dir, file))
            if file_num > max_file_num:
                to_move.sort()
                os.remove(to_move[0])
                to_move.pop(0)

            torch.save(state, save_dir + (f'/model_best_{epoch}.pth.tar'))

    def adjust_learning_rate(optimizer, epoch, start_lr):
        lr = start_lr * (0.5 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    BEST_test_MAE_PER_AVG = np.inf
    writer = SummaryWriter(f'{save_dir}/logs/CNN')
    START_EPOCH = 0

    model = model.float()
    for epoch in range(START_EPOCH, EPOCH):
        adjust_learning_rate(opt, epoch, LR)
    
        model.train()
        train_MAE = 0.
        train_MAE_PER = 0.
        for batch_idx, (b_i, b_v, b_y) in enumerate(loader()[1]):
            output = model(b_i, b_v)

            # print(output)
            # break    

            # MAE = torch.mean(torch.square(output-b_y))
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_MAE += loss.item()

        train_MAE_AVG = train_MAE/(batch_idx+1)

        model.eval()
        test_MAE = 0.
        test_MAE_PER = 0.
        with torch.no_grad():
            for batch_idx, (b_i, b_v, b_y) in enumerate(loader()[0]):
                test_output = model(b_i, b_v)

                test_loss_fn = nn.CrossEntropyLoss()
                test_loss = test_loss_fn(test_output, b_y)
                test_MAE += test_loss.item()

            test_MAE_AVG = test_MAE/(batch_idx+1)

            print_log(f'Epoch: {epoch}\t'
                      f'|train_MAE_AVG: {train_MAE_AVG:.8f}\t'
                      f'|test_MAE_AVG: {test_MAE_AVG:.8f}\t', log)

            is_best = test_MAE_AVG < BEST_test_MAE_PER_AVG

            if is_best:
                BEST_test_MAE_PER_AVG = min(
                    test_MAE_AVG, BEST_test_MAE_PER_AVG)
                print('BEST_test_MAE_PER_AVG: ', BEST_test_MAE_PER_AVG)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'BEST_test_MAE_PER_AVG': BEST_test_MAE_PER_AVG,
                    'optimizer': opt.state_dict()
                }, is_best, epoch + 1)

        writer.add_scalar('train_MAE_PER_AVG', train_MAE, epoch)
        writer.add_scalar('test_MAE_PER_AVG', test_MAE, epoch)

    writer.close()


for num in range(0, 1):
    train_once_data(num)
