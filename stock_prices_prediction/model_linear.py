import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime


class LinearStockModel(nn.Module):
    def __init__(self, num_features, num_securities):
        super(LinearStockModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 192)
        self.bn1 = nn.BatchNorm1d(num_securities)
        self.fc2 = nn.Linear(192, 64)
        self.bn2 = nn.BatchNorm1d(num_securities)
        self.fc3 = nn.Linear(64, num_securities)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        # out = F.dropout(out, p=0.1, training=self.training)
        out = self.relu(self.bn2(self.fc2(out)))
        # out = F.dropout(out, p=0.1, training=self.training)

        return self.fc3(out.view(-1))


data_frame = pd.read_csv("C:/Dev/data/DCOILWTICO.csv", parse_dates=["DATE"], sep=",")
data_frame.drop(
    data_frame[data_frame["DCOILWTICO"].str.endswith(".")].index, inplace=True
)
data_frame["date_year"] = data_frame["DATE"].dt.year - 2015
data_frame["date_mon"] = data_frame["DATE"].dt.month
data_frame["date_dom"] = data_frame["DATE"].dt.day
dates = data_frame["DATE"].values
data_frame.drop(["DATE"], axis=1, inplace=True)
prices = data_frame["DCOILWTICO"].astype(float).to_numpy()
data_frame.drop(["DCOILWTICO"], axis=1, inplace=True)
inputs = data_frame.to_numpy()

print(inputs.shape, prices.shape)

loss = nn.MSELoss()
learning_rate = 1e-3
epochs = 600
predict_days = 60

net = LinearStockModel(inputs.shape[1], 1).cuda()
l = int(inputs.shape[0] * 0.95)
train = inputs[0:l]
labels = prices[0:l]

validation = inputs[l:]
targets = prices[l:]

optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # , weight_decay=1e-1)

for epoch in range(epochs + 1):
    l = 0
    idx = 0
    for row in train:
        x = torch.FloatTensor(row.reshape((1, 1, -1))).cuda()
        y = torch.FloatTensor([labels[idx]]).cuda()
        idx += 1

        optimizer.zero_grad()
        y_pred = net(x)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()

    preds = []
    for row in validation:
        v = net(torch.FloatTensor(row.reshape((1, 1, -1))).cuda()).detach().item()
        preds.append(v)

    diff = targets - np.array(preds)

    print("Epoch {} of {}".format(epoch, epochs))
    print("Diff for best prediction (mean, stdev, min, max, loss, lr):")
    print(diff.mean(), diff.std(), diff.min(), diff.max(), l.item(), learning_rate)

    learning_rate *= 0.98
    if epoch % 100 == 0:
        learning_rate = 1e-3
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)


preds = []
for row in train:
    v = net(torch.FloatTensor(row.reshape((1, 1, -1))).cuda()).detach().item()
    preds.append(v)

for row in validation:
    v = net(torch.FloatTensor(row.reshape((1, 1, -1))).cuda()).detach().item()
    preds.append(v)

y = inputs[-1][0]
m = inputs[-1][1]
d = inputs[-1][2]
future = np.array([])
for _ in range(predict_days):
    d += 1
    if d > 29 or (m == 1 and d > 27):
        d = 0
        m += 1
        if m > 11:
            m = 0
            y += 1
    future = np.append(future, np.array([y, m, d]))
    dt = datetime.datetime(year=2015 + y, month=m + 1, day=d + 1)
    dates = np.append(dates, np.datetime64(dt))

future = future.reshape((-1, inputs.shape[1]))

for row in future:
    v = net(torch.FloatTensor(row.reshape((1, 1, -1))).cuda()).detach().item()
    preds.append(v)
    prices = np.append(prices, None)


plt.xlabel("Dates")
plt.ylabel("Prices")
plt.plot(dates, prices, "b")
plt.plot(dates, np.array(preds), "g")
plt.show()
