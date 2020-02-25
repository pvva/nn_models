import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


class DaRnnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, time_steps):
        # input size: number of underlying factors
        # hidden_size: dimension of the hidden state
        # time_steps: number of time steps
        super(DaRnnEncoder, self).__init__()
        self.input_size = input_size  # I
        self.hidden_size = hidden_size  # H
        self.time_steps = time_steps  # T

        self.attn_size = 2 * hidden_size + time_steps - 1  # A

        self.lstm_layer = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=1
        )
        self.attn_linear = nn.Linear(in_features=self.attn_size, out_features=1)

    def forward(self, input_data):
        # input_data: B x T-1 x I
        # B x T-1 x I
        input_weighted = torch.zeros(
            input_data.size(0), self.time_steps - 1, self.input_size
        ).cuda()
        # B x T-1 x I
        input_encoded = torch.zeros(
            input_data.size(0), self.time_steps - 1, self.hidden_size
        )
        # initial states for LSTM
        # 1 x B x H
        hidden = torch.zeros(1, input_data.size(0), self.hidden_size).cuda()
        cell = torch.zeros(1, input_data.size(0), self.hidden_size).cuda()
        for step in range(self.time_steps - 1):
            # concatenate the hidden states with each predictor

            # B x I x A
            x = torch.cat(
                (
                    hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                    cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                    input_data.permute(0, 2, 1),
                ),
                dim=2,
            )
            # calculate attention weights, B*I x 1 (B*I x A (after .view(-1,..) -> linear layer)
            x = self.attn_linear(x.view(-1, self.attn_size))

            # B x I
            attn_weights = F.softmax(x.view(-1, self.input_size), dim=0)
            # B x I, input data is B x T-1 x I
            weighted_input = torch.mul(attn_weights, input_data[:, step, :])
            # LSTM takes in 1 x B x I
            _, lstm_states = self.lstm_layer(
                weighted_input.unsqueeze(0), (hidden, cell)
            )
            hidden = lstm_states[0]  # 1 x B x H
            cell = lstm_states[1]  # 1 x B x H

            input_weighted[:, step, :] = weighted_input
            input_encoded[:, step, :] = hidden.squeeze()

        return input_weighted, input_encoded


class DaRnnDecoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, time_steps):
        super(DaRnnDecoder, self).__init__()

        self.time_steps = time_steps  # T
        self.encoder_hidden_size = encoder_hidden_size  # He
        self.decoder_hidden_size = decoder_hidden_size  # Hd
        self.attn_size = 2 * decoder_hidden_size + encoder_hidden_size  # A

        self.attn_layer = nn.Sequential(
            nn.Linear(self.attn_size, encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, 1),
        )
        self.lstm_layer = nn.LSTM(
            input_size=1, hidden_size=decoder_hidden_size, num_layers=1
        )
        self.fc = nn.Linear(encoder_hidden_size + 1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, 1)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: B x T-1 x He
        # y_history: B x T-1
        # 1 x B x Hd
        hidden = torch.zeros(1, input_encoded.size(0), self.decoder_hidden_size).cuda()
        cell = torch.zeros(1, input_encoded.size(0), self.decoder_hidden_size).cuda()
        context = None
        for step in range(self.time_steps - 1):
            # compute attention weights
            # B x T x A
            x = torch.cat(
                (
                    hidden.repeat(self.time_steps - 1, 1, 1).permute(1, 0, 2),
                    cell.repeat(self.time_steps - 1, 1, 1).permute(1, 0, 2),
                    input_encoded,
                ),
                dim=2,
            )
            # B * T-1
            x = F.softmax(
                self.attn_layer(x.view(-1, self.attn_size)).view(
                    -1, self.time_steps - 1
                ),
                dim=0,
            )
            # compute context
            # bmm: if input is a (B x N x M) tensor, mat2 is a (B x M x P) tensor, out will be a (B x N x P) tensor
            # x.unsqueeze(1) => B x 1 x T-1, input_encoded => B x T-1 x He, result => B x 1 x He => (squeeze) B x He
            context = torch.bmm(x.unsqueeze(1), input_encoded).squeeze(1)

            if step < self.time_steps - 1:
                # input is cat(B x He, B x 1) => B x He+1, result is B x 1
                t_y = self.fc(
                    torch.cat((context, y_history[:, step].unsqueeze(1)), dim=1)
                )
                # LSTM takes in 1 x B x 1
                _, lstm_output = self.lstm_layer(t_y.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0]  # 1 x B x Hd
                cell = lstm_output[1]  # 1 x B x Hd

        return self.fc_final(torch.cat((hidden[0], context), dim=1))


class DaRnn:
    def __init__(
        self,
        data_file,
        date_field,
        price_field,
        encoder_hidden_size=64,
        decoder_hidden_size=64,
        time_steps=10,
        learning_rate=1e-2,
        batch_size=128,
        parallel=True,  # if multiple GPUs
    ):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.__prepare_data(data_file, date_field, price_field)
        self.train_size = int(self.X.shape[0] * 0.9)
        # target data can be normalized, but in reality optimizer also works without normalisation
        # self.Y = self.Y - np.mean(self.Y[: self.train_size])
        self.learning_rate = learning_rate

        self.encoder = DaRnnEncoder(
            input_size=self.X.shape[1],
            hidden_size=encoder_hidden_size,
            time_steps=time_steps,
        ).cuda()
        self.decoder = DaRnnDecoder(
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            time_steps=time_steps,
        ).cuda()

        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

    def __prepare_data(self, data_file, date_field, price_field):
        data_frame = pd.read_csv(data_file, parse_dates=[date_field], sep=",")
        data_frame.drop(
            data_frame[data_frame[price_field].str.endswith(".")].index, inplace=True
        )
        data_frame.drop(
            data_frame[data_frame[price_field].str.startswith(".")].index, inplace=True
        )
        self.dates = data_frame[date_field].values
        data_frame.drop([date_field], axis=1, inplace=True)

        # list of prices
        self.prices = data_frame[price_field].astype(float).to_numpy()

        # Data size x Time steps
        self.X = np.array(
            [
                [self.prices[j + i] for i in range(self.time_steps)]
                for j in range(0, len(self.prices) - self.time_steps - 1, 1)
            ]
        )
        # Data size
        self.Y = np.array(
            [
                self.prices[j + self.time_steps]
                for j in range(0, len(self.prices) - self.time_steps - 1, 1)
            ]
        )

    def train(self, n_epochs=10):
        ilr = self.learning_rate
        loss_func = nn.MSELoss()

        for epoch in range(n_epochs):
            iter_loses = np.array([])
            # randomly permute indices to form generic non continuous batches
            perm_idx = np.random.permutation(self.train_size - self.time_steps)
            cIdx = 0
            while cIdx < self.train_size:
                batch_idx = perm_idx[cIdx : (cIdx + self.batch_size)]

                # B x T-1 x I
                input_data = np.zeros(
                    (len(batch_idx), self.time_steps - 1, self.X.shape[1])
                )
                # B x T-1
                y_history = np.zeros((len(batch_idx), self.time_steps - 1))
                # B
                y_target = self.Y[batch_idx + self.time_steps]

                for k in range(len(batch_idx)):
                    input_data[k, :, :] = self.X[
                        batch_idx[k] : (batch_idx[k] + self.time_steps - 1), :
                    ]
                    y_history[k, :] = self.Y[
                        batch_idx[k] : (batch_idx[k] + self.time_steps - 1)
                    ]

                loss = self.train_iteration(input_data, y_history, y_target, loss_func)
                iter_loses = np.append(iter_loses, loss)
                cIdx += self.batch_size

            e_loss_mean = np.mean(iter_loses).item()
            e_loss_std = np.std(iter_loses).item()
            print(
                "Epoch %d, loss mean: %3.3f, loss std: %3.3f, lr: %1.5f"
                % (epoch + 1, e_loss_mean, e_loss_std, self.learning_rate)
            )

            # adapt learning rate
            # saw pattern (slightly decreasing rate at every epoch and then every, say,
            # 100 epochs resetting it to original rate, works worse then gradual rate decrease
            if epoch % 100 == 0 and epoch > 0:
                self.learning_rate *= 0.98
                self.encoder_optimizer = optim.Adam(
                    self.encoder.parameters(), lr=self.learning_rate
                )
                self.decoder_optimizer = optim.Adam(
                    self.decoder.parameters(), lr=self.learning_rate
                )

    def train_iteration(self, x, y_history, y_target, loss_func):
        # x : B x T-1 x I
        # y_history : B x T-1
        # y_target: B
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.encoder(torch.from_numpy(x).float().cuda())
        y_pred = self.decoder(
            input_encoded, torch.from_numpy(y_history).float().cuda()
        ).flatten()

        y_true = torch.from_numpy(y_target).float().cuda()
        loss = loss_func(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def predict(self, on_train=False):
        if on_train:
            y_pred = np.zeros(self.train_size - self.time_steps + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_size)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            # B x T-1 x I
            inputs = np.zeros((len(batch_idx), self.time_steps - 1, self.X.shape[1]))
            # B x T-1
            y_history = np.zeros((len(batch_idx), self.time_steps - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    inputs[j, :, :] = self.X[
                        range(batch_idx[j], batch_idx[j] + self.time_steps - 1), :
                    ]
                    y_history[j, :] = self.Y[
                        range(batch_idx[j], batch_idx[j] + self.time_steps - 1)
                    ]
                else:
                    inputs[j, :, :] = self.X[
                        range(
                            batch_idx[j] + self.train_size - self.time_steps,
                            batch_idx[j] + self.train_size - 1,
                        ),
                        :,
                    ]
                    y_history[j, :] = self.Y[
                        range(
                            batch_idx[j] + self.train_size - self.time_steps,
                            batch_idx[j] + self.train_size - 1,
                        )
                    ]

            y_history = torch.from_numpy(y_history).float().cuda()
            _, input_encoded = self.encoder(torch.from_numpy(inputs).float().cuda())
            y_pred[i : (i + self.batch_size)] = (
                self.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            )
            i += self.batch_size

        return y_pred

    # prev_history comes as array of values
    def predict_future(self, prev_history, batches=1):
        y_pred = np.zeros(batches * self.batch_size)

        s = self.batch_size - 1 + self.time_steps * 2 - 1
        h_s = prev_history

        i = 0
        while i < len(y_pred):
            # B x T
            h_x = np.array(
                [
                    [h_s[j + i] for i in range(self.time_steps)]
                    for j in range(len(h_s)-s, len(h_s)-self.time_steps, 1)
                ]
            )
            # B
            h_y = np.array(
                [
                    h_s[j + self.time_steps]
                    for j in range(len(h_s)-s, len(h_s)-self.time_steps, 1)
                ]
            )

            batch_idx = np.array(range(len(y_pred)))[i : (i + self.batch_size)]
            inputs = np.zeros((len(batch_idx), self.time_steps - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.time_steps - 1))

            for j in range(self.batch_size):
                inputs[j, :, :] = h_x[
                    range(j, j + self.time_steps - 1), :
                ]
                y_history[j, :] = h_y[
                    range(j, j + self.time_steps - 1)
                ]

            y_history = torch.from_numpy(y_history).float().cuda()
            _, input_encoded = self.encoder(torch.from_numpy(inputs).float().cuda())
            p_v = self.decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
            y_pred[i : (i + self.batch_size)] = p_v
            i += self.batch_size
            h_s = np.append(h_s, p_v)

        return y_pred

    def predict_and_plot(self, future_batches=1):
        predictions_t = np.array(self.predict(True))
        predictions_v = np.array(self.predict(False))
        predictions_f = np.array(self.predict_future(self.prices, future_batches))

        last_date = pd.to_datetime(self.dates[-1])
        y = last_date.year
        m = last_date.month
        d = last_date.day

        for _ in range(future_batches * self.batch_size):
            d += 1
            if d > 29 or (m == 1 and d > 27):
                d = 0
                m += 1
                if m > 11:
                    m = 0
                    y += 1
            dt = datetime.datetime(year=y, month=m + 1, day=d + 1)
            self.dates = np.append(self.dates, np.datetime64(dt))
            self.prices = np.append(self.prices, None)

        bv = len(predictions_t)
        plt.xlabel("Dates")
        plt.ylabel("Prices")
        plt.plot(self.dates, self.prices, "blue", label="True")

        plt.plot(
            self.dates[self.time_steps : bv + self.time_steps],
            predictions_t,
            "green",
            label="Predicted T",
        )
        bv += self.time_steps
        plt.plot(
            self.dates[bv : bv + len(predictions_v)],
            predictions_v,
            "red",
            label="Predicted V",
        )
        bv += len(predictions_v)
        plt.plot(
            self.dates[bv : bv + len(predictions_f)],
            predictions_f,
            "magenta",
            label="Predicted F",
        )

        plt.legend(loc="upper left")
        plt.show()


model = DaRnn(
    data_file="./data/DCOILWTICO.csv",
    date_field="DATE",
    price_field="DCOILWTICO",
    batch_size=96,
    time_steps=16,
    learning_rate=1e-3,
    encoder_hidden_size=128,
    decoder_hidden_size=128,
)
model.train(300)
model.predict_and_plot(future_batches=2)
