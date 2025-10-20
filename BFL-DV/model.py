import torch
import torch.nn as nn

class CNNLSTMNet(nn.Module):
    def __init__(self, input_len=336, num_classes=2, lstm_hidden=64, lstm_layers=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        # 经过两次池化，序列长度变为 input_len // 4
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=False
        )
        self.fc1 = nn.Linear((input_len // 4) * lstm_hidden, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

        self.dp = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: (batch, 1, 336)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch, 32, input_len//4)
        x = x.permute(0, 2, 1)  # (batch, seq_len, feature) for LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden)
        lstm_out = lstm_out.contiguous().view(x.size(0), -1)  # flatten
        x = self.fc1(lstm_out)
        x = self.relu3(x)
        x = self.dp(x)
        x = self.fc2(x)
        return x 