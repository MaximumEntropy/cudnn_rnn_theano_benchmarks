"""PyTorch RNN Test."""
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import time
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--network",
    help="network type rnn/lstm/gru",
    required=True
)
parser.add_argument(
    "-d",
    "--depth",
    help="num layers",
    type=int,
    required=True
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    help="batch size",
    required=True
)
parser.add_argument(
    "-o",
    "--hidden",
    type=int,
    help="hidden dim",
    required=True
)
parser.add_argument(
    "-t",
    "--seq_len",
    type=int,
    help="time steps",
    required=True
)
args = parser.parse_args()
network_type = args.network
depth = args.depth
batch_size = args.batch_size
hidden_dim = args.hidden
seq_len = args.seq_len
num_passes = 1000


class StackedLSTM(nn.Module):
    """Deep LSTM."""

    def __init__(
        self,
        input_size,
        rnn_size,
        num_layers,
    ):
        """Initialize params."""
        super(StackedLSTM, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.layers = []

        for i in range(num_layers):
            layer = LSTM(
                input_size, rnn_size
            )
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the layer."""
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            output, (h_1_i, c_1_i) = layer(input, (h_0, c_0))
            print output.size()
            input = output
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    """Deep GRU."""

    def __init__(
        self,
        input_size,
        rnn_size,
        num_layers,
    ):
        """Initialize params."""
        super(StackedGRU, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.layers = []

        for i in range(num_layers):
            layer = GRU(
                input_size, rnn_size
            )
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the layer."""
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            output, (h_1_i,) = layer(input, (h_0,), ctx)
            input = output

            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, (h_1,)


class LSTM(nn.Module):
    r"""A long short-term memory (LSTM) cell."""

    def __init__(self, input_size, hidden_size):
        """Initialize params."""
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)

            return hy, cy

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden


class GRU(nn.Module):
    r"""A gated recurrent (GRU) cell."""

    def __init__(self, input_size, hidden_size):
        """Initialize params."""
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.input_weights = nn.Linear(input_size, 3 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx = hidden
            gi = self.input_weights(input)
            gh = self.hidden_weights(hx)

            i_r, i_i, i_n = gi.chunk(3, 1)
            h_r, h_i, h_n = gh.chunk(3, 1)

            resetgate = F.sigmoid(i_r + h_r)
            inputgate = F.sigmoid(i_i + h_i)
            newgate = F.tanh(i_n + resetgate * h_n)

            hy = newgate + inputgate * (hidden - newgate)

            return hy

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden

x_val = Variable(torch.randn(seq_len, batch_size, hidden_dim)).cuda()
y_val = Variable(torch.randn(seq_len, batch_size, hidden_dim)).cuda()

h0_val = Variable(torch.randn(depth, batch_size, hidden_dim)).cuda()
c0_val = Variable(torch.randn(depth, batch_size, hidden_dim)).cuda()

start = time.time()

if network_type == 'cudnn_lstm':

    rnn = nn.LSTM(
        hidden_dim,
        hidden_dim,
        depth
    ).cuda()

elif network_type == 'cudnn_gru':

    rnn = nn.GRU(
        hidden_dim,
        hidden_dim,
        depth
    ).cuda()

if network_type == 'lstm':

    rnn = StackedLSTM(
        hidden_dim,
        hidden_dim,
        depth
    ).cuda()

elif network_type == 'gru':

    rnn = StackedGRU(
        hidden_dim,
        hidden_dim,
        depth
    ).cuda()

print "Setup : compile + forward/backward x 1"
print "--- %s seconds" % (time.time() - start)

if network_type == 'cudnn_lstm' or network_type == 'lstm':
    num_processed = num_passes * batch_size

    if network_type == 'lstm':
        h0_val = h0_val.squeeze()
        c0_val = c0_val.squeeze()

    start = time.time()
    for i in xrange(0, num_passes):
        output, (_, _) = rnn(x_val, (h0_val, c0_val))
    end = time.time()
    print "Forward:"
    print "--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (
        num_processed,
        end - start,
        num_processed / (end - start),
        (end - start) / num_processed
    )

    start = time.time()
    for i in xrange(0, num_passes):
        output, (_, _) = rnn(x_val, (h0_val, c0_val))
        loss = ((y_val - output) ** 2).mean()
        loss.backward()
    end = time.time()
    print "Forward + Backward:"
    print "--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (
        num_processed,
        end - start,
        num_processed / (end - start),
        (end - start) / num_processed
    )

if network_type == 'cudnn_gru' or network_type == 'gru':
    num_processed = num_passes * batch_size

    if network_type == 'gru':
        h0_val = h0_val.squeeze()

    start = time.time()
    for i in xrange(0, num_passes):
        output, (_, _) = rnn(x_val, (h0_val,))
    end = time.time()
    print "Forward:"
    print "--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (
        num_processed,
        end - start,
        num_processed / (end - start),
        (end - start) / num_processed
    )

    start = time.time()
    for i in xrange(0, num_passes):
        output, (_, _) = rnn(x_val, (h0_val,))
        loss = ((y_val - output) ** 2).mean()
        loss.backward()
    end = time.time()
    print "Forward + Backward:"
    print "--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (
        num_processed,
        end - start,
        num_processed / (end - start),
        (end - start) / num_processed
    )
