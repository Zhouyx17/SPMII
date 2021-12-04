import torch
import torch.nn as nn
import torch.nn.functional as F

class textCNN(nn.Module):
    def __init__(self,vocab_size, embed_dim, class_num, kernel_num, kernel_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_size])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(kernel_size)*kernel_num, class_num)
    
    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]
        
        x = torch.cat(x,1)
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits