import torch 
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F 
#import matplotlib.pyplot as plt 
  
  
class IndelNet(torch.nn.Module): 
  def __init__(self, n_features, hidden_sizes, n_output): 
    super(IndelNet, self).__init__()  
    '''
    self.layers = []
    self.layers.append(nn.Linear(n_features, hidden_sizes[0]))
    for i in range(len(hidden_sizes) - 1):
      print("add hidden layer:({} x {})".format(hidden_sizes[i], hidden_sizes[i+1]))
      self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
    '''
    self.hidden0 = nn.Linear(n_features, hidden_sizes[0])
    assert(len(hidden_sizes) == 5)
    self.hidden1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
    self.hidden2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
    self.hidden3 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
    self.hidden4 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
    self.predict = nn.Linear(hidden_sizes[-1], n_output) # output layer
    #self.predict = nn.Linear(hidden_sizes[0], n_output) # output layer
    self.dropout = nn.Dropout(p = 0.5)
  
  def forward(self, x): 
    '''
    for i in range(len(self.layers)):
      x = F.relu(self.layers[i](x))
    '''
    #net = Net(n_feature, [140, 160, 170, 100, 10] , 2)
    #print("----------------print out input--------------")
    act_f = F.leaky_relu
    #act_f = torch.tanh
    x = act_f(self.hidden0(x)) 
    x = self.dropout(x)
    x = act_f(self.hidden1(x)) 
    x = self.dropout(x)
    x = act_f(self.hidden2(x))
    x = self.dropout(x)
    x = act_f(self.hidden3(x))
    x = self.dropout(x)
    x = act_f(self.hidden4(x))
    x = self.predict(x) 
    #print("6\n", x)
    return x
    #return F.softmax(x, dim = 1)

  def initialize_weights(self):
    print("initializeing net work")
    for m in self.modules():
      print(m)
      if isinstance(m, nn.Linear):
        # print(m.weight.data.type())
        # input()
        # m.weight.data.fill_(1.0)
        #nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('Leaky Relu'))
        nn.init.xavier_uniform_(m.weight)

class Net(torch.nn.Module):  #---snv net---#
  def __init__(self, n_features, hidden_sizes, n_output): 
    super(Net, self).__init__()  
    '''
    self.layers = []
    self.layers.append(nn.Linear(n_features, hidden_sizes[0]))
    for i in range(len(hidden_sizes) - 1):
      print("add hidden layer:({} x {})".format(hidden_sizes[i], hidden_sizes[i+1]))
      self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
    '''
    self.hidden0 = nn.Linear(n_features, hidden_sizes[0])
    assert(len(hidden_sizes) == 5)
    self.hidden1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
    self.hidden2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
    self.hidden3 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
    self.hidden4 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
    self.predict = nn.Linear(hidden_sizes[-1], n_output) # output layer
    #self.predict = nn.Linear(hidden_sizes[0], n_output) # output layer
    self.dropout = nn.Dropout(p = 0.5)
  
  def forward(self, x): 
    '''
    for i in range(len(self.layers)):
      x = F.relu(self.layers[i](x))
    '''
    #print("----------------print out input--------------")
    act_f = F.leaky_relu
    #act_f = torch.tanh
    x = act_f(self.hidden0(x)) 
    x = self.dropout(x)
    x = act_f(self.hidden1(x)) 
    x = self.dropout(x)
    x = act_f(self.hidden2(x))
    x = self.dropout(x)
    x = act_f(self.hidden3(x))
    x = self.dropout(x)
    x = act_f(self.hidden4(x))
    x = self.predict(x) 
    #print("6\n", x)
    return x
    #return F.softmax(x, dim = 1)

  def initialize_weights(self):
    print("initializeing net work")
    for m in self.modules():
      print(m)
      if isinstance(m, nn.Linear):
        # print(m.weight.data.type())
        # input()
        # m.weight.data.fill_(1.0)
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
        #nn.init.xavier_uniform_(m.weight, gain = nn.init.calculate_gain('tanh'))
