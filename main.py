import torch
import torch.nn as nn
import numpy as np
import collections
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from DeepCCAModels import DeepCCA

import time

from evaluate import evalutateTranslation

from sklearn.model_selection import KFold
import random
torch.set_default_tensor_type(torch.DoubleTensor)


class Solver():
    def __init__(self, model, linear_cca, outdim_size, batch_size, learning_rate, reg_par, device=torch.device('cpu'), momentum = 0):
        self.model = nn.DataParallel(model)
        self.model.to(device)
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=reg_par)
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size


    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        x1.to(self.device)
        x2.to(self.device)

        data_size = x1.size(0)
        train_loss = 0
        train_losses = []
        epoch = 1
        while True:
            it = 0
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2)
                #print(loss)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            if train_loss - np.mean(train_losses) <= 1e-3 or epoch >= 100:
                #print('epoch num is: ', epoch, ' trainloss didnt improve, train loss is:    ', train_loss)
                it += 1
                if it >= 5 or epoch >= 100:
                    print('convergenced, loss is:    ', train_loss)
                    break
                
            else:
                #print('epoch num is: ', epoch, 'improvement is: ', train_loss - np.mean(train_losses))
                train_loss = np.mean(train_losses)
            #print('epoc num is: ',epoch, ' current loss is: ', train_loss)
            epoch = epoch+1 

        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)
            self.train_linear_cca(outputs[0], outputs[1])
              
    
    def test(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                #print("Linear CCA started!")
                outputs = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs
            else:
                return np.mean(losses)

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs

def load_data(path):
    
    vol_src,emb_src = read_data(path[0])
    vol_trg, emb_trg = read_data(path[1])
    
    #build test dictionary
    src_word2ind = {word: i for i, word in enumerate(vol_src)}
    trg_word2ind = {word: i for i, word in enumerate(vol_trg)}
    
    src_indices_train,trg_indices_train = load_dictionary(path[2], src_word2ind, trg_word2ind)
    src_indices_test,trg_indices_test = load_dictionary(path[3], src_word2ind, trg_word2ind)

    emb_src = np.array(emb_src, dtype="float32")
    emb_trg = np.array(emb_trg, dtype="float32")

    return vol_src, emb_src, vol_trg, emb_trg, src_indices_train, trg_indices_train, src_indices_test, trg_indices_test
        
def read_data(path):
    f = open(path, encoding="utf-8", errors='surrogateescape')
    dtype = np.float32
    header = f.readline().split(' ')
    count = np.int(header[0])
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype)
    for i in range(count):
        word, vec = f.readline().split(' ', 1)
        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
    return words,matrix

def load_dictionary(path,src_word2ind,trg_word2ind):
    src_indices = []
    trg_indices = []

    f = open(path, encoding="utf-8", errors='surrogateescape')
    for line in f:
        src_word,trg_word = line.split()
        try:
            src_ind = src_word2ind[src_word]
            trg_ind = trg_word2ind[trg_word]
            src_indices.append(src_ind)
            trg_indices.append(trg_ind)
        except KeyError:
            print("OOV words occurs")
            
    return src_indices, trg_indices

def mean_center(x):
    avg = np.mean(x, axis=0)
    x -= avg
    return x

def length_normalize(x):
    norms = np.sqrt(np.sum(x**2, axis=1))
    norms[norms == 0] = 1
    x /= norms[:, np.newaxis]
    return x




path = ["/media/jzhao1/498032e5-c16b-4e25-b7b7-5186838340e2/data/CCA/embeddings/en.emb.txt",
        "/media/jzhao1/498032e5-c16b-4e25-b7b7-5186838340e2/data/CCA/embeddings/de.emb.txt",
        "/media/jzhao1/498032e5-c16b-4e25-b7b7-5186838340e2/data/CCA/dictionaries/en-de.train.txt", 
        "/media/jzhao1/498032e5-c16b-4e25-b7b7-5186838340e2/data/CCA/dictionaries/en-de.test.txt"]

vol_src,x,vol_trg,y,src_indices_train,trg_indices_train,src_indices_test, trg_indices_test = load_data(path)


x = mean_center(x)
y = mean_center(y)
x = length_normalize(x)
y = length_normalize(y)
   
x = torch.Tensor(x)
y = torch.Tensor(y)
    
x_dic = x[src_indices_train]
y_dic = y[trg_indices_train]

# shuffle the input data
num_list = [i for i in range(len(x_dic))]
random.shuffle(num_list)
x_dic = x_dic[num_list]
y_dic = y_dic[num_list]


src_indices_train = np.array(src_indices_train)
trg_indices_train = np.array(trg_indices_train)
src_indices_train = src_indices_train[num_list].tolist()
trg_indices_train = trg_indices_train[num_list].tolist()


def dcca_validate(args):
    
    np.set_printoptions(suppress=True, linewidth=2000, precision=15)
    print(args)
    outdim_size = args['outdim_size']
    input_shape1 = 300
    input_shape2 = 300
    # number of layers with nodes in each one
    layer_src = []
    
    src_layer_size = args['src_layer_size']
    src_H1_num = args['src_H1_num']
    src_H2_num = args['src_H2_num']
    src_H3_num = args['src_H3_num']
    src_H4_num = args['src_H4_num']
    
  #  if src_layer_size == 1:
   #     layer_src.append(src_H1_num)
    if src_layer_size == 2:
        layer_src.append(src_H1_num)
        layer_src.append(src_H2_num)
    elif src_layer_size == 3:
        layer_src.append(src_H1_num)
        layer_src.append(src_H2_num)
        layer_src.append(src_H3_num)
    elif src_layer_size == 4:
        layer_src.append(src_H1_num)
        layer_src.append(src_H2_num)
        layer_src.append(src_H3_num)
        layer_src.append(src_H4_num)


    layer_trg = []
    trg_layer_size =args['trg_layer_size']
    trg_H1_num = args['trg_H1_num']
    trg_H2_num = args['trg_H2_num']
    trg_H3_num = args['trg_H3_num']
    trg_H4_num = args['trg_H4_num']

    #if trg_layer_size == 1:
     #   layer_trg.append(trg_H1_num)
    if trg_layer_size == 2:
        layer_trg.append(trg_H1_num)
        layer_trg.append(trg_H2_num)
    elif trg_layer_size == 3:
        layer_trg.append(trg_H1_num)
        layer_trg.append(trg_H2_num)
        layer_trg.append(trg_H3_num)
    elif trg_layer_size == 4:
        layer_trg.append(trg_H1_num)
        layer_trg.append(trg_H2_num)
        layer_trg.append(trg_H3_num)
        layer_trg.append(trg_H4_num)

    
    layer_src.append(outdim_size)
    layer_trg.append(outdim_size)
    
    # the parameters for training the network
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    momentum = args['momentum']
    reg_par = args['reg_par']
    parameterset = []

    parameterset.append(src_layer_size)
    parameterset.append(trg_layer_size)
    
    parameterset.append(src_H1_num)
    parameterset.append(src_H2_num)
    parameterset.append(src_H3_num)
    parameterset.append(src_H4_num)

    parameterset.append(trg_H1_num)
    parameterset.append(trg_H2_num)
    parameterset.append(trg_H3_num)
    parameterset.append(trg_H4_num)
    
    
    parameterset.append(batch_size)
    parameterset.append(momentum)
    parameterset.append(learning_rate)
    parameterset.append(args['r1'])
    parameterset.append(args['r2'])
    parameterset.append(args['r3'])
    parameterset.append(args['r4'])
    parameterset.append(args['reg_par'])
    parameterset = np.array(parameterset)
    

    para = 'corresponding para is: ' + str(parameterset) + '\n'
    f = open(args['name'], 'a')
    f.write(para)
    f.close()
    
    
    
    use_all_singular_values = False
    apply_linear_cca = True
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca(r3 = args['r3'], r4 = args['r4'])

    bst_val_loss = 0 
    kf = KFold(n_splits=2)

    acc_all = []
    loss_all = []

    for train, test in kf.split(x_dic):      
        acc = 0
        X_train = torch.Tensor(x_dic[train])
        Y_train = torch.Tensor(y_dic[train])

        src_indices_train_ = np.array(src_indices_train)
        trg_indices_train_ = np.array(trg_indices_train)
        
        validation_idx_src = src_indices_train_[test].tolist()
        validation_idx_trg = trg_indices_train_[test].tolist()

        model = DeepCCA(layer_src, layer_trg, input_shape1,
                    input_shape2, outdim_size, use_all_singular_values, device=device, r1=args['r1'], r2 = args['r2']).double()   
        
        solver = Solver(model, l_cca, outdim_size, batch_size,
                    learning_rate, reg_par, device=device, momentum = momentum)

        solver.fit(X_train, Y_train)
        loss, output = solver.test(x, y, apply_linear_cca)
        
        acc = evalutateTranslation(output[0], output[1], vol_src, vol_trg, validation_idx_src, validation_idx_trg, "nn", 5000)
        print('acc is: ', acc)
        
        acc_all.append(acc)        
        loss_all.append(loss)

        
        del solver
        del model
    
    
    acc_avg = float(np.mean(acc_all))
    acc_std = float(np.std(acc_all))

    loss_avg = float(np.mean(loss_all))
    loss_std = float(np.std(loss_all))
    print('eval finished, the avg acc is: ', acc_avg)
    result_txt = 'acc_avg is: ' + str(acc_avg) + '  acc_std is: ' + str(acc_std)  + '  loss_avg is: ' + str(loss_avg) + '  loss_std is: ' + str(loss_std) +'\n'
    f = open(args['name'], 'a')
    f.write(result_txt)
    f.close()


    raw_txt = str(args) + '\n'
    f = open(args['raw'], 'a')
    f.write(raw_txt)
    f.close()
    return  {'loss': 1-acc_avg, 'status': STATUS_OK}


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import argparse
import hyperopt



if __name__ == '__main__':
        
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description="global argument group")
    parser.add_argument("--name", default = 'result0', type = str,help="using cuda or not")
    parser.add_argument('--raw', default= 'raw0', type=str, help = 'raw')
    args = parser.parse_args()
    name = args.name
    raw = args.raw
    fspace = {
        
        'outdim_size' : hp.choice('outdim_size', [300]),
    
        'src_layer_size' : hp.choice('src_layer_size', [2, 3, 4]),
        'trg_layer_size' : hp.choice('trg_layer_size', [2, 3, 4]),
        
        'src_H1_num' : hp.choice('src_H1_num', [400, 512, 640, 1024, 2048, 4096]),
        'src_H2_num' : hp.choice('src_H2_num', [640, 1024, 2048, 4096]),
        'src_H3_num' : hp.choice('src_H3_num', [640, 1024, 2048, 4096]),
        'src_H4_num' : hp.choice('src_H4_num', [640, 1024, 2048, 4096]),

        'trg_H1_num' : hp.choice('trg_H1_num', [400, 512, 640, 1024, 2048, 4096]),
        'trg_H2_num' : hp.choice('trg_H2_num', [640, 1024, 2048, 4096]),
        'trg_H3_num' : hp.choice('trg_H3_num', [640, 1024, 2048, 4096]),
        'trg_H4_num' : hp.choice('trg_H4_num', [640, 1024, 2048, 4096]),
        

        'batch_size' : hp.choice('batch_size', [500]),
        #'momentum': hp.choice('momentum', [0.9]),
        'momentum' : hp.uniform('momentum', 0, 1),
        'learning_rate' : hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
        #'learning_rate' : hp.choice('learning_rate', [0.01]),
        #'r1' : hp.choice('r1', [1e-8]),
        #'r2' : hp.choice('r2', [1e-8]),
        #'r3' : hp.choice('r3', [1e-4]),
        #'r4' : hp.choice('r4', [1e-4]),
        'r1' : hp.loguniform('r1', np.log(1e-9), np.log(1e-5)),
        'r2' : hp.loguniform('r2', np.log(1e-9), np.log(1e-5)),
        'r3' : hp.loguniform('r3', np.log(1e-5), np.log(1e-3)),
        'r4' : hp.loguniform('r4', np.log(1e-5), np.log(1e-3)),
        #'reg_par' : hp.choice('reg_par', [1e-4]),
        'reg_par' : hp.loguniform('reg_par', np.log(1e-6), np.log(1e-4)),
        
        'name': hp.choice('name', [name]),
        'raw' : hp.choice('raw', [raw])
    }
    trials = Trials()
    best = fmin(fn=dcca_validate, space=fspace, algo=hyperopt.rand.suggest, max_evals=500, trials=trials)
    
