import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.datasets import TUDataset

##
from RB_Tree_Sampler.RB_tree import Free_func, test_main
##
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=3, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)


args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

#print(args)





num_training = int(len(dataset) * 0.1)
num_val = int(len(dataset) * 0.8)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

if training_set[1].x[0][0] != training_set[1].x[0][1]:
    for i in range(0, len(training_set) ):
        training_set[i].x[0][0] = i
        training_set[i].x[0][1] = i

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory = True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)


model = Model(args).to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


####
def Sim(x_1: torch.Tensor, x_2: torch.Tensor):
    Sim_dist = torch.norm(x_1 - x_2, p=2)
    return Sim_dist.clamp(min=1e-12).sqrt()  # for numerical stability
    #-torch.mean(x_1 - x_norm) + torch.log(torch.mean(torch.exp(x_2 - x_norm)))
####
def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    #model.load_state_dict(torch.load('HGP-SL/modelinicial/1.pth'))
    
    model.load_state_dict(torch.load('HGP-SL/modelinicial/100my_file.npy') )
    model.train()
    Already_in = []##
    more_terms = 0##
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            #data_ = data.to_data_list
            #print("data_1:",data[1])
            #print("data:",data)
            
            optimizer.zero_grad()
            data = data.to(args.device)
            out, out_Feature = model(data)
            loss = 0
            Node = [] 
            Node_num = []
            Re_space = []
            ###
            test_0 = 0
            test_1 = 0
            test_2 = 0
            ###
            for j in range(0, len(data.y)):
                #print("d_1:", data.y[])
                data_y = torch.unsqueeze(data.y, 1) 
                loss_ = F.nll_loss(torch.unsqueeze(out[j], 0), data_y[j])
                if loss_ > 1:
                    if data[j].x[0][0] not in Already_in:
                        #print('yes_1')
                        Already_in.append(data[j].x[0][0])
                        node_ = [out_Feature[j].to("cpu"), data[j].to("cpu"), data_y[j].to("cpu"), loss_.to("cpu"), i]
                        Node.append(node_)
                        Node_num.append(data[j].x[0][0])
                        
                        if loss > 2:
                            test_1 += 1
                        
                        test_0 += 1
                    elif loss_ < 2:
                        Re_space.append(node_)
                if loss_ < 0.1:
                    if data[j].x[0][0] not in Already_in:
                        #print("yes_2")
                        Already_in.append(data[j].x[0][0])
                        node_ = [out_Feature[j].to("cpu"), data[j].to("cpu"), data_y[j].to("cpu"), loss_.to("cpu"), i]
                        Node.append(node_)
                        Node_num.append(data[j].x[0][0])
                        test_2 += 1
                loss += loss_
            #print("test_1:", test_1)
            #print("test_2:", test_2)
            #print("test_1/0:", test_1/test_0)
            
            #data_ = to_data_list(data)
            loss.backward()
            optimizer.step()
            loss_origin = loss
            #test_main(Node, Node_num)
            ####
            Node_ = [] 
            Node_num_ = []
            data_list_review = test_main(Node, Node_num)
            more_terms += len(data_list_review)
            print(more_terms)
            if len(data_list_review) != 0:
                Review_loader = DataListLoader(data_list_review, batch_size=64, shuffle=False)
                
                for j, data in enumerate(Review_loader):
                    #Conservatism_or_Adventurism = 0#1 for Conservatism, 2 for Adventurism and 0 for not decided
                    optimizer.zero_grad()
                    data = data.to(args.device)
                    out, out_Feature = model(data)
                    loss = 0
                    AnchorLoss = 0
                    VentureGain = 0
                    for k in range(0, len(data.y)):
                        #print("d_1:", data.y[])
                        data_y = torch.unsqueeze(data.y, 1) 
                        loss_ = F.nll_loss(torch.unsqueeze(out[k], 0), data_y[k])
                        loss += loss_
                        '''if k in AnchorList:
                            if loss_ > 0.10:
                                AnchorLoss += Loss_
                                node_ = Node[Node_num.index(data[k].x[0][0])]
                                if Sim(node_[0], out_Feature[k] ) > 0.1:
                                    Node_.append(node_)
                                    Node_num_.append((node_[1][0][0])
                                    #Conservatism_or_Adventurism = 1                                    
                        elif k in VentureList:
                            if loss_ < BaseLoss - 0.1:
                                if loss < 0.8:
                                    VentureGain += BaseLoss - loss_
                                    node_ = Node[Node_num.index(data[k].x[0][0])]
                                    if Sim(node_[0], out_Feature[k] ) > 0.1 and True:
                                                     ##Conservatism_or_Adventurism != 1:
                                        Node_.append(node_)
                                        Node_num_.append((node_[1][0][0])
                                        ##Conservatism_or_Adventurism -= 1
                    ##if Conservatism_or_Adventurism == 0:
                      ##  break'''
                        
                #data_list_review, AnchorList, VentureList, BaseLoss = test_main(Node, Node_num)
                    
                    loss.backward()
                    optimizer.step()
            ####
            
            loss_train += loss_origin.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(loss_val)
        ####
        
        store = 'HGP-SL/modelinicial/' + str(100)+'my_file.npy'          
        ##torch.save(model.state_dict(), store)
        
        ####
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data)[0]
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    # Model training
    Free_func()####
    best_model = train()
    # Restore best model for test set
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    test_acc, test_loss = compute_test(test_loader)
    print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
