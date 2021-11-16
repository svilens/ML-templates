import pandas as pd

df = pd.read_csv('data.csv')

label = 'class'
num_features = len([x for x in df.columns if x != label])
num_classes = len(df[label].unique())

X = df.drop(label, axis=1).values
y = df[label].values

# Scale
from sklearn.preprocessing import MinMaxScaler
X_transformed = MinMaxScaler().fit_transform(X)

# Train-Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed,
    y,
    test_size=0.30,
    random_state=0
)


###########
# PyTorch #
###########

import torch
import torch.nn as nn
import torch.utils.data as td

torch.manual_seed(0)

# Create a dataset and loader for the training data and labels
train_x = torch.Tensor(X_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=False, num_workers=1)

# Create a dataset and loader for the test data and labels
test_x = torch.Tensor(X_test).float()
test_y = torch.Tensor(y_test).long()
test_ds = td.TensorDataset(test_x,test_y)
test_loader = td.DataLoader(test_ds, batch_size=20, shuffle=False, num_workers=1)


##################
# Neural Network #
##################

num_hidden_layer_nodes = 10

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer_input = nn.Linear(num_features, num_hidden_layer_nodes)
        self.layer_hidden = nn.Linear(num_hidden_layer_nodes, num_hidden_layer_nodes)
        self.layer_output = nn.Linear(num_hidden_layer_nodes, num_classes)
    def forward(self, x):
        x = torch.relu(self.layer_input(x))
        x = torch.relu(self.layer_hidden(x))
        x = torch.relu(self.layer_output(x))
        return x

model = NeuralNet()
print(model)


#########
# Train #
#########

def train(model, data_loader, optimizer):
    # Set the model to training mode
    model.train()
    train_loss = 0
    
    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        #feedforward
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()

        # backpropagate
        loss.backward()
        optimizer.step()

    #Return average loss
    avg_loss = train_loss / (batch+1)
    print(f'Training set: Average loss: {avg_loss:.6f}')
    return avg_loss


def test(model, data_loader):
    # Switch the model to evaluation mode (so we don't backpropagate)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        batch_count = 0
        for batch, tensor in enumerate(data_loader):
            batch_count += 1
            data, target = tensor
            # Get the predictions
            out = model(data)

            # calculate the loss
            test_loss += loss_criteria(out, target).item()

            # Calculate the accuracy
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss/batch_count
    print(
        f'Validation set: Average loss: {avg_loss:.6f},\
            Accuracy: {correct}/{len(data_loader.dataset)} \
            ({100. * correct / len(data_loader.dataset):.0f}%)\n')
    
    # return average loss for the epoch
    return avg_loss


# Specify the loss criteria (we'll use CrossEntropyLoss for multi-class classification)
loss_criteria = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()

# track metrics for each epoch in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 50 epochs
epochs = 50
for epoch in range(1, epochs + 1):
    print(f'Epoch: {epoch}')
    train_loss = train(model, train_loader, optimizer)
    test_loss = test(model, test_loader)
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)


from matplotlib import pyplot as plt

plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()