
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialDropout(nn.Dropout2d):
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = super(SpatialDropout, self).forward(x)  
        x = x.permute(0, 2, 1)  
        return x
    

class GRU_model(nn.Module):
    def __init__(
        self,
        args,
        pred_len=68
    ):
        super(GRU_model, self).__init__()
        self.pred_len = pred_len

        self.embedding = nn.Embedding(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim)
        self.cnn_layer = nn.Conv1d(in_channels=16, out_channels=args.embedding_dim, kernel_size=5, padding=5//2)
        
        self.embedding_dropout = SpatialDropout(0.3)

        self.gru = nn.GRU(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.hidden_layers,
            dropout=args.dropout,
            bidirectional=True,
            batch_first=True
        )

        self.linear = nn.Linear(args.hidden_size * 2, 5)

    def forward(self, seqs):
        seqs = seqs.permute(0, 2, 1)
        embed = self.cnn_layer(seqs)
        embed = self.embedding_dropout(embed)
        reshaped = embed.permute(0, 2, 1) #torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        output, hidden = self.gru(reshaped)
        turncated = output[:, :self.pred_len, :]
        out = self.linear(turncated)
        
        return out
