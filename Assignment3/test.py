import torch
from utils import *
import argparse

if __name__ == '__main__':
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
                        help="Weight deay if we apply some.") 
    parser.add_argument("--total_epoch", default=500, type=int,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    args, model = getVisionTransformers_model(args)
    train_loader, test_loader = get_loader(args)


    model.load_state_dict(torch.load('./checkpoint_500/ckpt.pth')['net'])
    print(model)
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))
            
            image, label = image.to(device), label.to(device)

            output, s = model(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    print()
    print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
