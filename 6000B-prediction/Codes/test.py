from numpy import genfromtxt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np

input_size = 57
hidden_size1 = 64
hidden_size2 = 64
num_classes = 2
num_epochs = 256
batch_size = 1380
learning_rate = 0.001

data = genfromtxt('testdata.csv', delimiter=',')
print(data.shape)
test_dataset = Data.TensorDataset(data_tensor = torch.from_numpy(data).float(), target_tensor = torch.from_numpy(data).float())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										   batch_size=batch_size,
										   shuffle=False)

class Net(nn.Module):
	def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
		super(Net, self).__init__()
		self.fc1 = nn.Sequential(
			nn.Linear(input_size, hidden_size1),
			nn.ReLU())
		self.fc2 = nn.Sequential(
			nn.Linear(hidden_size1, hidden_size2),
			nn.ReLU())
		self.fc3 = nn.Sequential(
			nn.Linear(hidden_size2, num_classes))

	def forward(self, x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		return out

net = Net(input_size, hidden_size1, hidden_size2, num_classes)
net.load_state_dict(torch.load('app2l1.pkl'))
net.eval()

predictions = np.zeros((1380,1))
for data in test_loader:
	inputs, label = data
	inputs = Variable(inputs)
	outputs = net(inputs)
	_, preds = torch.max(outputs.data, 1)
	print(preds)
	predictions = preds.numpy()

f = open("project1_20478966.csv", "w+")
for i in range(1380):
	f.write(str(predictions[i]) + "\n")
f.close()
