# Cell
import numpy as np
import torch
import torch.nn as nn

from .base_model import PyMADModel
from ..utils.utils import de_unfold

# Cell
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.dropout = dropout

    def forward(self, inputs, hidden):
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)
        gates = (torch.matmul(inputs, self.weight_ih.t()) + self.bias_ih +
                         torch.matmul(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

# Cell
class ResLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMCell, self).__init__()
        self.register_buffer('input_size', torch.Tensor([input_size]))
        self.register_buffer('hidden_size', torch.Tensor([hidden_size]))
        self.weight_ii = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_ic = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ii = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ic = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(1 * hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(1 * hidden_size))
        self.weight_ir = nn.Parameter(torch.randn(hidden_size, input_size))
        self.dropout = dropout

    def forward(self, inputs, hidden):
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)

        ifo_gates = (torch.matmul(inputs, self.weight_ii.t()) + self.bias_ii +
                                  torch.matmul(hx, self.weight_ih.t()) + self.bias_ih +
                                  torch.matmul(cx, self.weight_ic.t()) + self.bias_ic)
        ingate, forgetgate, outgate = ifo_gates.chunk(3, 1)

        cellgate = torch.matmul(hx, self.weight_hh.t()) + self.bias_hh

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        ry = torch.tanh(cy)

        if self.input_size == self.hidden_size:
            hy = outgate * (ry + inputs)
        else:
            hy = outgate * (ry + torch.matmul(inputs, self.weight_ir.t()))
        return hy, (hy, cy)

# Cell
class ResLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ResLSTMCell(input_size, hidden_size, dropout=0.)

    def forward(self, inputs, hidden):
        inputs = inputs.unbind(0)
        outputs = []
        for i in range(len(inputs)):
                out, hidden = self.cell(inputs[i], hidden)
                outputs += [out]
        outputs = torch.stack(outputs)
        return outputs, hidden

# Cell
class AttentiveLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(AttentiveLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        attention_hsize = hidden_size
        self.attention_hsize = attention_hsize

        self.cell = LSTMCell(input_size, hidden_size)
        self.attn_layer = nn.Sequential(nn.Linear(2 * hidden_size + input_size, attention_hsize),
                                        nn.Tanh(),
                                        nn.Linear(attention_hsize, 1))
        self.softmax = nn.Softmax(dim=0)
        self.dropout = dropout

    def forward(self, inputs, hidden):
        inputs = inputs.unbind(0)
        outputs = []

        for t in range(len(inputs)):
            # attention on windows
            hx, cx = (tensor.squeeze(0) for tensor in hidden)
            hx_rep = hx.repeat(len(inputs), 1, 1)
            cx_rep = cx.repeat(len(inputs), 1, 1)
            x = torch.cat((inputs, hx_rep, cx_rep), dim=-1)
            l = self.attn_layer(x)
            beta = self.softmax(l)
            context = torch.bmm(beta.permute(1, 2, 0),
                                inputs.permute(1, 0, 2)).squeeze(1)
            out, hidden = self.cell(context, hidden)
            outputs += [out]
        outputs = torch.stack(outputs)
        return outputs, hidden

# Cell
class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dilations, dropout=0, cell_type='GRU', batch_first=False):
        super(DRNN, self).__init__()

        self.dilations = dilations
        self.cell_type = cell_type
        self.batch_first = batch_first

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        elif self.cell_type == "ResLSTM":
            cell = ResLSTMLayer
        elif self.cell_type == "AttentiveLSTM":
            cell = AttentiveLSTMLayer
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size,
                                                       hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            hidden = torch.zeros(batch_size * rate, hidden_size,
                                 dtype=dilated_inputs.dtype,
                                 device=dilated_inputs.device)
            hidden = hidden.unsqueeze(0)

            if self.cell_type in ['LSTM', 'ResLSTM', 'AttentiveLSTM']:
                hidden = (hidden, hidden)

        dilated_outputs, hidden = cell(dilated_inputs, hidden) # compatibility hack

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        iseven = (n_steps % rate) == 0

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2),
                                 dtype=inputs.dtype,
                                 device=inputs.device)
            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs


class _RNN(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 n_t: int, cell_type: str, dilations: list, state_hsize: int, add_nl_layer: bool):
        super(_RNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.n_t = n_t
        self.cell_type = cell_type
        self.dilations = dilations
        self.state_hsize = state_hsize
        self.add_nl_layer = add_nl_layer
        self.layers = len(dilations)

        layers = []
        for grp_num in range(len(self.dilations)):
            if grp_num == 0:
                input_size = self.input_size + (self.input_size + self.output_size)*self.n_t
            else:
                input_size = self.state_hsize
            layer = DRNN(input_size,
                         self.state_hsize,
                         n_layers=len(self.dilations[grp_num]),
                         dilations=self.dilations[grp_num],
                         cell_type=self.cell_type)
            layers.append(layer)

        self.rnn_stack = nn.Sequential(*layers)

        if self.add_nl_layer:
            self.MLPW  = nn.Linear(self.state_hsize, self.state_hsize)

        self.adapterW  = nn.Linear(self.state_hsize, self.output_size)

    def forward(self, Y: torch.Tensor, X: torch.Tensor):
        if self.n_t >0:
            input_data = torch.cat((Y, X), -1)
        else:
            input_data = Y

        for layer_num in range(len(self.rnn_stack)):
            residual = input_data
            output, _ = self.rnn_stack[layer_num](input_data)
            if layer_num > 0:
                output += residual
            input_data = output

        if self.add_nl_layer:
            input_data = self.MLPW(input_data)
            input_data = torch.tanh(input_data)

        input_data = self.adapterW(input_data)
        input_data = input_data.transpose(0,1) #change to bs, n_windows

        return input_data

class RNN(PyMADModel):
    def __init__(self,
                 window_size,
                 window_step,
                 input_size,
                 output_size,
                 sample_freq,
                 n_t,
                 cell_type,
                 dilations,
                 state_hsize,
                 add_nl_layer,
                 random_seed,
                 device=None):
        super(RNN, self).__init__(window_size, window_step, device)
        
        assert sample_freq==output_size, 'Sample freq and output_size must be equal (for now)'

        # RNN
        self.input_size = input_size
        self.output_size = output_size
        self.sample_freq = sample_freq
        self.n_t = n_t
        self.cell_type = cell_type
        self.dilations = dilations
        self.state_hsize = state_hsize
        self.add_nl_layer = add_nl_layer
        self.random_seed = random_seed

        # Generator
        torch.manual_seed(random_seed)

        # Model
        self.model = _RNN(input_size=input_size, output_size=output_size, n_t=n_t,
                          cell_type=cell_type, dilations=dilations, state_hsize=state_hsize,
                          add_nl_layer=add_nl_layer)

        self.training_type = 'sgd'

    def parse_batch(self, batch):
        Y = batch['Y']
        X = batch['X']
        mask = batch['mask']

        # Expects [batch_size, n_features, temporal] -> [batch_size*n_features, temporal]
        Y = Y.reshape(-1, Y.shape[-1])
        mask = mask.reshape(-1, Y.shape[-1])

        # Rolling windows
        right_padding = self.output_size - (Y.shape[-1]-self.output_size*(1 + (Y.shape[-1]-self.output_size)//self.sample_freq))
        padder = torch.nn.ConstantPad1d(padding=(self.input_size, right_padding), value=0)
        Y = padder(Y)
        mask = padder(mask)

        Y = Y.unfold(dimension=-1, size=self.input_size+self.output_size, step=self.sample_freq)
        mask = mask.unfold(dimension=-1, size=self.input_size+self.output_size, step=self.sample_freq)
        Y = Y.transpose(0,1) # [n_features, n_windows, time] -> [n_windows, n_features, time]
        mask = mask.transpose(0,1) #  [n_features, n_windows, time] -> [n_windows, n_features, time]

        if self.n_t>0:
            X = X.unfold(dimension=-1, size=self.input_size+self.output_size, step=self.sample_freq)
            X = X.transpose(1,2) # batch, n_windows, n_t, time
            X = X.flatten(start_dim=2)
            X = X.transpose(0,1) # n_windows, batch, n_t*time

        return Y, X, mask

    def forward(self, input):

        # Creates rolling windows for RNN
        Y, X, mask = self.parse_batch(batch=input)

        # Hide with mask
        Y = Y * mask

        # Input - output split for each window
        insample_Y = Y[:, :, :self.input_size]
        outsample_Y = Y[:, :, self.input_size:]
        outsample_mask = mask[:,:, self.input_size:]

        outsample_Y = outsample_Y.transpose(0,1) # [n_features, n_windows, time] -> [n_windows, n_features, time]
        outsample_mask = outsample_mask.transpose(0,1) # [n_features, n_windows, time] -> [n_windows, n_features, time]

        # Forward pass
        y_hat = self.model(Y=insample_Y, X=X)

        return outsample_Y, y_hat, outsample_mask

    def training_step(self, input):

        self.model.train()

        # Forward
        Y, Y_hat, mask = self.forward(input=input)

        # MAE as loss (for now)
        loss = torch.mean(torch.abs(Y-Y_hat))
        # loss = torch.mean(mask*torch.abs(Y-Y_hat))

        return loss

    def eval_step(self, x):
        self.model.eval()
        loss = self.training_step(x)
        return loss

    def window_anomaly_score(self, input, return_detail: bool=False):

        self.model.eval()

        batch_size, n_features, window_size = input['Y'].shape
        assert batch_size == 1, 'Batch size must be 1 (for now)'

        # Forward
        Y, Y_hat, mask = self.forward(input=input)
        
        Y = Y.reshape(n_features, -1)[:,:window_size] # to [n_features, n_time]
        Y_hat = Y_hat.reshape(n_features, -1)[:,:window_size]  # to [n_features, n_time]
        mask = mask.reshape(n_features, -1)[:,:window_size]  # to [n_features, n_time]

        # Add mask dimension
        Y = Y[None, :, :]
        Y_hat = Y_hat[None, :, :]
        mask = mask[None, :, :]
        
        if return_detail:
            return torch.abs(Y-Y_hat)*mask
        else:
            return torch.mean(np.abs(Y-Y_hat)*mask, dim=1)

    def final_anomaly_score(self, input, return_detail: bool=False):
        
        # Average anomaly score for each feature per timestamp
        anomaly_scores = de_unfold(windows=input, window_step=self.window_step)

        if return_detail:
            return anomaly_scores
        else:
            anomaly_scores = torch.mean(anomaly_scores, dim=0)
            return anomaly_scores
