import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from model import *
from utils import *
import ml_collections
import argparse
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import random
import torch.optim as optim


#用测试集评估模型的训练好坏
def eval(args, model, test_loader, epoch):
    start = time.time()
    eval_loss=0.0
    total_acc=0.0
    model.eval()

    for i,batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits, _ = model(x)  # model返回的是（bs,num_classes）和weight
            batch_loss=loss_function(logits,y)
            # 记录误差
            eval_loss += batch_loss.item()
            # 记录准确率
            _, preds = logits.max(1)
            num_correct = (preds==y).sum().item()
            total_acc += num_correct
    
    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        eval_loss / len(test_loader.dataset),
        total_acc / len(test_loader.dataset),
        finish - start
    ))


    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', eval_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', total_acc / len(test_loader.dataset), epoch)
    return total_acc / len(test_loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100",
                        help="Which downstream task.")
    parser.add_argument("--img_size", default=224, type=int,help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=5e-4, type=float,
                        help="Weight deay if we apply some.")  # 一开始设为0结果就过拟合了
    parser.add_argument("--total_epoch", default=200, type=int,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    log_dir = 'runs'  # tensorboard log dir
    milestones = [60, 120, 160]

    args, model = getVisionTransformers_model(args)

    print("load dataset.........................")
    #加载数据
    train_loader, test_loader = get_loader(args)
    # Prepare optimizer and scheduler
    optimizer = optim.SGD(model.parameters(),
                            lr=args.learning_rate,
                            momentum=0.9,
                            weight_decay=args.weight_decay)
    loss_function = torch.nn.CrossEntropyLoss()
    # learning rate decay
    warmup_epoch = 5
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch)

    print("training.........................")

    # use tensorboard
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, args.dataset, datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')))
    best_acc = 0.0

    for i in range(1, args.total_epoch + 1):
        if i > warmup_epoch:
            train_scheduler.step()

        start = time.time()
        model.train()

        cut = Cutout(n_holes=1, length=16)
        train_loss=0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            loss = model(images, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # iteration number 
            n_iter = i * len(train_loader) + step + 1
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch = i,
                trained_samples = step * args.train_batch_size + len(images),
                total_samples = len(train_loader.dataset)
            ))      

            #update training loss for each iteration
            writer.add_scalar('Train/loss', loss.item(), n_iter)
            
            if i <= warmup_epoch:
                warmup_scheduler.step()
                
        finish = time.time()

        print('epoch {} training time consumed: {:.2f}s'.format(i, finish - start))


        # 每训练一个epoch,用当前训练的模型对验证集进行测试
        eval_acc = eval(args, model, test_loader, i)


        # 保存最佳的模型参数
        if best_acc < eval_acc:
            state = {
                'net': model.state_dict(),
                'acc': eval_acc,
                'epoch': i
            }
            if not os.path.isdir('checkpoint_200'):
                os.mkdir('checkpoint_200')
            torch.save(state, './checkpoint_200/ckpt.pth')
            best_acc = eval_acc

        train_scheduler.step()
    
    writer.close()

