import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utility import (
    load_object
)

# Defne a NN model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        super(Net, self).__init__()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

def main(fileName = "results/sensitivity_analysis_10_57_34_10_04_2024"):
    
    parameters_list = load_object(fileName + "/Data", "param_values")
    returns_timeseries = load_object(fileName + "/Data", "returns_timeseries")
    
    input_size = len(returns_timeseries[0])
    hidden_size = 50
    output_size = len(parameters_list[0])

    #Prepare the data for the neural network
    X = torch.tensor(np.asarray(returns_timeseries)).float()
    y = torch.tensor(parameters_list).float()

    #We split into training and testing data
    train_size = int(0.8 * len(X))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    #Creta the NN model
    model = Net(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Train the model and keep the loss to plot it later
    epochs = 2000
    train_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_X)
        loss = criterion(output, train_y)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch} loss: {loss.item()}')

    #Plot the loss
    plt.plot(train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    #Test the model
    model.eval()
    with torch.no_grad():
        test_output = model(test_X)
        loss = criterion(test_output, test_y)
        print(f'Test loss: {loss.item()}')

    #Save the NN model 
    torch.save(model.state_dict(), fileName + "/Data/NN_model.pt")
    plt.show()

if __name__ == '__main__':
    filename = "results/surrogate_model_11_28_59_26_04_2024"
    main(filename)


