import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)

    def forward(self, input):
        """
        
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out
    
class Encoder(nn.Module):
    def __init__(self, input_dim=64, lstm_hidden_dim=256, output_dim=8):
        super(Encoder, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        
    def forward(self, x, lstm_hiddens):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, updated_hiddens = self.lstm(x.view(x.size()[0], 1, -1), lstm_hiddens) 
        # Take the last output from LSTM
        output = self.fc(lstm_out[:, 0, :])  # output shape: (batch_size, output_dim)
        return output, updated_hiddens

class Decoder(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=6):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.fc1(x)  # x shape: (batch_size, hidden_dim)
        x = self.relu(x)
        output = self.fc2(x)
        return output
