"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Training & Validation
"""
import numpy as np 
import argparse, cv2
import logging
import time
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
import torch.utils.tensorboard as tensorboard
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

import utils
from model.model import Mini_Yception
from dataset import create_train_dataloader, create_val_dataloader, create_test_dataloader
from utils import visualize_confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

cudnn.benchmark = True
cudnn.enabled = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=15, help="training batch size")
    parser.add_argument('--tensorboard', type=str, default='checkpoint/tensorboard', help='path log dir of tensorboard')
    parser.add_argument('--logging', type=str, default='checkpoint', help='path of logging')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='optimizer weight decay')
    parser.add_argument('--datapath', type=str, default='data', help='root path of dataset')
    parser.add_argument('--test_datapath', type=str, default='data', help='root path of test dataset')
    parser.add_argument('--pretrained', type=str,default='checkpoint/model_weights/train_original.pth.tar',help='load checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from pretrained path specified in prev arg')
    parser.add_argument('--savepath', type=str, default='checkpoint/model_weights', help='save checkpoint path')    
    parser.add_argument('--savefreq', type=int, default=1, help="save weights each freq num of epochs")
    parser.add_argument('--logdir', type=str, default='checkpoint/logging', help='logging')
    parser.add_argument("--lr_patience", default=40, type=int)
    parser.add_argument('--evaluate', action='store_true', help='evaluation only')
    parser.add_argument('--mode', type=str, default='val', choices=['val','test', 'train'], help='dataset type for evaluation only')
    parser.add_argument('--age_mode', action='store_true', help='age mode')

    args = parser.parse_args()
    return args
# ======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
# logging
logging.basicConfig(
format='[%(message)s',
level=logging.INFO,
handlers=[logging.FileHandler(args.logdir + "_y_" +
                              args.datapath.split("/")[-1] + "_" +
                              str(args.batch_size) + "_" +
                              str(args.lr) + "_" +
                              str(args.lr_patience) + "_" +
                              str(args.weight_decay), mode='w'), logging.StreamHandler()])
# tensorboard
writer = tensorboard.SummaryWriter(args.tensorboard)

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.RandomEqualize(p=1),
                                # transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor()])

def main():
    # ========= dataloaders ===========
    if args.datapath == "data":
        train_dataloader = create_train_dataloader(root=args.datapath, batch_size=args.batch_size)
        test_dataloader = create_val_dataloader(root=args.datapath, batch_size=args.batch_size)
    else:
        trainDataset = datasets.ImageFolder(args.datapath + "/Train", transform=transform)
        print(trainDataset.class_to_idx)
        train_dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)

        if args.test_datapath == "data":
            test_dataloader = create_val_dataloader(root="data", batch_size=args.batch_size)
        else:
            testDataset = datasets.ImageFolder(args.test_datapath + "/Test", transform=transform)
            test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=args.batch_size, shuffle=True)

    # train_dataloader, test_dataloader = create_CK_dataloader(batch_size=args.batch_size)
    start_epoch = 0
    # ======== models & loss ==========
    if args.age_mode:
        mini_xception = Mini_Yception(5)
    else:
        mini_xception = Mini_Yception()

    loss = nn.CrossEntropyLoss()
    # ========= load weights ===========
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.pretrained, map_location=device)
        mini_xception.load_state_dict(checkpoint['mini_xception'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        print(f'\tLoaded checkpoint from {args.pretrained}\n')
        time.sleep(1)
    else:
        print("******************* Start training from scratch *******************\n")
        time.sleep(2)

    if args.evaluate:
        if args.test_datapath == "data":
            if args.mode == 'test':
                test_dataloader = create_test_dataloader(args.test_datapath, batch_size=args.batch_size)
            elif args.mode == 'val':
                test_dataloader = create_val_dataloader(args.test_datapath, batch_size=args.batch_size)
            else:
                test_dataloader = create_train_dataloader(args.test_datapath, batch_size=args.batch_size)
        else:
            if args.mode == 'val':
                testDataset = datasets.ImageFolder(args.test_datapath + "/Test", transform=transform)
                test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=args.batch_size, shuffle=True)
            elif args.mode == 'train':
                testDataset = datasets.ImageFolder(args.test_datapath + "/Train", transform=transform)
                test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=args.batch_size, shuffle=True)

        validate(mini_xception, loss, test_dataloader, 0)
        return

    # =========== optimizer =========== 
    # parameters = mini_xception.named_parameters()
    # for name, p in parameters:
    #     print(p.requires_grad, name)
    # return
    optimizer = torch.optim.Adam(mini_xception.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, verbose=True)
    # ========================================================================
    for epoch in range(start_epoch, args.epochs):
        # =========== train / validate ===========
        train_loss = train_one_epoch(mini_xception, loss, optimizer, train_dataloader, epoch)
        val_loss, accuracy, percision, recall = validate(mini_xception, loss, test_dataloader, epoch)
        scheduler.step(val_loss)
        val_loss, accuracy, percision, recall = round(val_loss,3), round(accuracy,3), round(percision,3), round(recall,3)
        logging.info(f"\ttraining epoch={epoch} .. train_loss={train_loss}")
        logging.info(f"\tvalidation epoch={epoch} .. val_loss={val_loss}")
        logging.info(f'\tAccuracy = {accuracy*100} % .. Percision = {percision*100} % .. Recall = {recall*100} % \n')
        time.sleep(2)
        # ============= tensorboard =============
        writer.add_scalar('train_loss',train_loss, epoch)
        writer.add_scalar('val_loss',val_loss, epoch)
        writer.add_scalar('percision',percision, epoch)
        writer.add_scalar('recall',recall, epoch)
        writer.add_scalar('accuracy',accuracy, epoch)
        # ============== save model =============
        if epoch % args.savefreq == 0:
            checkpoint_state = {
                'mini_xception': mini_xception.state_dict(),
                "epoch": epoch
            }
            savepath = os.path.join(args.savepath, "y_"+f'{epoch}' + "_" +
                                                   args.datapath.split("/")[-1] + "_" +
                                                   str(args.batch_size) + "_" +
                                                   str(args.lr) + "_" +
                                                   str(args.lr_patience) + "_" +
                                                   str(args.weight_decay)
                                                   + '.pth.tar')
            torch.save(checkpoint_state, savepath)
            print(f'\n\t*** Saved checkpoint in {savepath} ***\n')
            time.sleep(2)
    writer.close()

def train_one_epoch(model, criterion, optimizer, dataloader, epoch):
    model.train()
    model.to(device)
    losses = []

    for images, labels in tqdm(dataloader):

        images = images.to(device) # (batch, 1, 48, 48)
        labels = labels.to(device) # (batch,)
        
        emotions = model(images)
        # from (batch, 7, 1, 1) to (batch, 7)
        emotions = torch.squeeze(emotions)
        # print(emotions)
        # print(labels,'\n')

        if len(labels) == 1:
            labels = labels[0]

        loss = criterion(emotions, labels)
        losses.append(loss.cpu().item())
        print(f'training @ epoch {epoch} .. loss = {round(loss.item(),3)}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # images = images.squeeze().cpu().detach().numpy()
        # cv2.imshow('f', images[0])
        # cv2.waitKey(0)

    return round(np.mean(losses).item(),3)


def validate(model, criterion, dataloader, epoch):
    model.eval()
    model.to(device)
    losses = []

    total_pred = []
    total_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            mini_batch = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)

            emotions = model(images)
            emotions = torch.squeeze(emotions)
            emotions = emotions.reshape(mini_batch, -1)

            loss = criterion(emotions, labels)

            losses.append(loss.cpu().item())

            # # ============== Evaluation ===============
            # index of the max value of each sample (shape = (batch,))
            _, indexes = torch.max(emotions, axis=1)
            # print(indexes.shape, labels.shape)
            total_pred.extend(indexes.cpu().detach().numpy())
            total_labels.extend(labels.cpu().detach().numpy())

            print(f'validation loss = {round(loss.item(),3)}')

        val_loss = np.mean(losses).item()
        percision = precision_score(total_labels, total_pred, average='macro')
        recall = recall_score(total_labels, total_pred, average='macro')
        accuracy = accuracy_score(total_labels, total_pred)

        val_loss, accuracy, percision, recall = round(val_loss,3), round(accuracy,3), round(percision,3), round(recall,3)    
        print(f'Val loss = {val_loss} .. Accuracy = {accuracy} .. Percision = {percision} .. Recall = {recall}')

        if args.evaluate:
            conf_matrix = confusion_matrix(total_labels, total_pred, normalize='true')
            print('Confusion Matrix\n', conf_matrix)
            visualize_confusion_matrix(conf_matrix)

        return val_loss, accuracy, percision, recall

if __name__ == "__main__":
    main()
    # total_labels = [0, 1, 2, 0]
    # total_pred =   [0, 2, 1, 2]
    # avg = None
    # avg = 'macro'
    # percision = precision_score(total_labels, total_pred, average=avg)
    # recall = recall_score(total_labels, total_pred, average=avg)
    # accuracy = accuracy_score(total_labels, total_pred)
    # print(percision, recall, accuracy)
