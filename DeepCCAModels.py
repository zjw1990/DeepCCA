import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            # output layer
            if l_id == len(layer_sizes) - 2:
                
                n_in = layer_sizes[l_id]
                n_out = layer_sizes[l_id+1]
                layers.append(
                    nn.Linear(n_in, n_out),
                )
                

                w_data = layers[l_id].weight.data
                b_data = layers[l_id].bias.data

                w_output = torch.from_numpy(np.random.normal(0, 0.1, size= (n_out, n_in)))
                b_output = torch.from_numpy(np.random.normal(0, 0.1, size= n_out))
                
                layers[l_id].weight.data = w_output
                layers[l_id].bias.data = b_output
            
            # hidden layers
            else:

                n_in = layer_sizes[l_id]
                n_out = layer_sizes[l_id+1]

                layers.append(nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.ReLU(),
                ))
                
                w_data = layers[l_id][0].weight.data
                b_data = layers[l_id][0].bias.data
                
                w = torch.from_numpy(np.random.normal(0, 0.1, size= (n_out, n_in)))
                b = torch.from_numpy(np.random.normal(0, 0.1, size= n_out))
                
                layers[l_id][0].weight.data = w
                layers[l_id][0].bias.data = b

        self.layers = nn.ModuleList(layers)

 

    def forward(self, x):
        for layer in self.layers:

            x = layer(x)

        return x


class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu'), r1=0, r2=0):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()

        self.loss = cca_loss(outdim_size, use_all_singular_values, device, r1, r2).loss
        #self.loss = nn.NLLLoss(w)

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2
