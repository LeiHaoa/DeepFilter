import torch
import torchvision

# load model
from nn_net import Net
import data_loader

VarType = "SNV"
n_feature = data_loader.FVC_INDEL_FEATURES if VarType == "INDEL" else data_loader.FVC_SNV_FEATURES


model = Net(n_feature, [140, 160, 170, 100, 10] , 2)
#model = Net(3, 2)#自己定义的网络模型
mnm = "checkpoint_fastvc_20-10-06-20-26-58_ecpch10"
model_dict = torch.load("./workspace/out/models/{}.pth".format(mnm))
model.load_state_dict(model_dict["state_dict"])#保存的训练模型
model.eval()#切换到eval（）

example = torch.rand(1, n_feature)
traced_script_module = torch.jit.trace(model, example)
#traced_script_module.save("model.pt")

traced_script_module.save("{}.pt".format(mnm))
#net = Net(n_feature, [140, 160, 170, 100, 10] , 2)
#model = net.model.cpu().eval()

#mnm = "checkpoint_fastvc_20-10-06-20-26-58_ecpch10"
#model = torch.load("./workspace/out/models/{}.pth".format(mnm))
#device = torch.device('cpu')
#
#example = torch.rand(1, n_feature)
#summary(model, input_size = (1, n_feature))
#
## Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
#traced_script_module = torch.jit.trace(model, example)
#traced_script_module.save("{}.pt".format(mnm))
