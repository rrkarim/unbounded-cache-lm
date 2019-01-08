import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.maks = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(1)
        input_size = context.size(1)

        attn = torch.bmm(otput, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        mix = torch.bmm(attn, context)

        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2*hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

