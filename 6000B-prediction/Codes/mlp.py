from numpy import genfromtxt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import os

# Parameters and Hyper Parameters
input_size = 57
hidden_size1 = 64
hidden_size2 = 64
num_classes = 2
num_epochs = 256
batch_size = 128
learning_rate = 0.001

data = genfromtxt('traindata.csv', delimiter=',')
labels = genfromtxt('trainlabel.csv', delimiter=',')

input_data = data[:3000] # train
test_input = data[3000:] # test
output_data = labels[:3000] # train
test_output = labels[3000:] # test

train_dataset = Data.TensorDataset(
	data_tensor = torch.from_numpy(input_data).float(),
	target_tensor = torch.from_numpy(output_data).long())

test_dataset = Data.TensorDataset(
	data_tensor = torch.from_numpy(test_input).float(),
	target_tensor = torch.from_numpy(test_output).long())

print(torch.from_numpy(output_data).int())
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										   batch_size=batch_size,
										   shuffle=False)

#Make a dictionary defining training and validation sets
dataloders = dict()
dataloders['train'] = train_loader
dataloders['val'] = test_loader

dataset_sizes = {'train': 3000, 'val': 220}
use_gpu = torch.cuda.is_available()

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

def train_model(model, criterion, optimizer, num_epochs):
	f = open("Iterations.txt", "w+")
	best_model_wts = model.state_dict()
	best_val_acc = 0.0
	best_train_acc = 0.0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train(True)  # Set model to training mode
			else:
				model.train(False)  # Set model to evaluate mode
			running_loss = 0.0
			running_corrects = 0
			# Iterate over data.
			for data in dataloders[phase]:
				# get the inputs
				inputs, label = data
				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					labels = Variable(label.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(label)
				# zero the parameter gradients
				optimizer.zero_grad()
				# forward
				outputs = model(inputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					optimizer.step()
				# statistics
				running_loss += loss.data[0]
				running_corrects += torch.sum(preds == label)
			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects / dataset_sizes[phase]
			#Print it out Loss and Accuracy and in the file torchvision
			print('{} Loss: {:.8f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			f.write('{} Loss: {:.8f} Accuracy: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))
			# deep copy the model
			if phase == 'val' and epoch_acc > best_val_acc:
				best_val_acc = epoch_acc
				best_model_wts = model.state_dict()
			if phase == 'train' and epoch_acc > best_train_acc:
				best_train_acc = epoch_acc
				best_model_wts = model.state_dict()
	f.close()
	print('Best val Acc: {:4f}'.format(best_val_acc))
	model.load_state_dict(best_model_wts)
	return model, best_train_acc, best_val_acc


net = Net(input_size, hidden_size1, hidden_size2, num_classes)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
if use_gpu:
	model_ft, train_acc, test_acc = train_model(net.cuda(), criterion, optimizer, num_epochs)
else:
	model_ft, train_acc, test_acc = train_model(net, criterion, optimizer, num_epochs)

torch.save(model_ft.state_dict(), 'save.pkl')
