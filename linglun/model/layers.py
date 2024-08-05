import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def positional_encoding(seq_len, d_model):
    """
    Args:
        seq_len (int): sequence length
        d_model (int): model dimension
    Returns:
        pos_enc (torch.Tensor): positional encoding, [:, seq_len, d_model]
    """
    pos = torch.arange(seq_len).unsqueeze(1) # [seq_len, 1]
    i = torch.arange(d_model // 2).unsqueeze(0) # [1, d_model//2]
    div_term = 1 / torch.pow(10000, (2 * i) / d_model)
    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 1::2] = torch.cos(pos * div_term)
    return pos_enc

def relative_scaled_dot_product_attention(Q, K, V, Srel, mask=None):
    """
    Args:
        Q (torch.Tensor): query, [batch_size, num_head, seq_len, d_k]
        K (torch.Tensor): key, [batch_size, num_head, seq_len, d_k]
        V (torch.Tensor): value, [batch_size, num_head, seq_len, d_k]
        Srel (torch.Tensor): relative positional encoding, [batch_size, num_head, seq_len, seq_len]
        mask (torch.Tensor): mask, [batch_size, num_head, seq_len, seq_len]
    Returns:
        output (torch.Tensor): output, [batch_size, num_head, seq_len, d_k]
    """
    d_k = Q.size(-1)
    QKt = torch.matmul(Q, K.transpose(-2, -1))
    scores =  (QKt + Srel) / math.sqrt(d_k) # [batch_size, num_head, seq_len, seq_len]
    if mask is not None:
        scores = scores + mask.masked_fill(mask == 1, float('-inf'))

    attention = F.softmax(scores, dim=-1) # [batch_size, num_head, seq_len, seq_len]
    output = torch.matmul(attention, V) # [batch_size, num_head, seq_len, d_k]
    return output

def skew(qet):
    L = qet.shape[-1] # get L 
    padded = F.pad(qet, (1, 0), "constant", float('-inf')) # 1. Pad a dummy column vector of length L before the leftmost column.
    srel = padded.reshape(-1, L+1, L) # 2. Reshape the matrix to have shape (L +1,L)
    srel = srel[:, 1:] # 3. slice the matrix to remove the first row
    srel = srel.reshape(*qet.shape)
    return srel

def get_relative_positional_encoding(embeddings, seq_len, d_model):
    """
    initial relative positional embedding matrix is arange from -max_len + 1 to 0 with shape [max_len, d_model]
    E = [e0, e1, e2, ..., emax_len-1], where e0 represents the relative position embedding of most distant token to the left
    In addtion, the paper is considering the casual attention, so the relative positional encoding matrix should be padded to the right
    We need to create a relative positional encoding matrix with shape [seq_len, d_model] by padding the embeddings
    """

    E = embeddings # embedding matrix [max_len, d_model] arange from -max_len + 1 to 0
    max_len = E.num_embeddings
    pad_len = max(seq_len - max_len, 0)
    if pad_len > 0:
        # use inf padding
        pad = torch.full((pad_len, d_model), float('-inf'), device=E.weight.device)
        # use zero padding
        # pad = torch.zeros(pad_len, d_model, device=E.weight.device)
        E_weights = torch.cat([pad, E.weight], dim=0)
    else: 
        E_weights = E.weight
    return E_weights




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_rel_attention, bias=True, batch_first=False) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_head = num_heads
        self.max_rel_attention = max_rel_attention
        self.bias = bias
        self.batch_first = batch_first

        assert d_model % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.d_h = self.d_model // self.num_head # depth of each head

        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=self.bias) # query
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=self.bias) # key
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=self.bias) # value

        # initialize the relative positional embedding matrix
        self.E = nn.Embedding(max_rel_attention, self.d_h) # relative positional embedding matrix

        self.fc = nn.Linear(self.d_model, self.d_model, bias=self.bias) # final linear layer

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q (torch.Tensor): query, [batch_size, seq_len, d_model]
            K (torch.Tensor): key, [batch_size, seq_len, d_model]
            V (torch.Tensor): value, [batch_size, seq_len, d_model]
            mask (torch.Tensor): mask, [batch_size, seq_len, seq_len]
        Returns:
            output (torch.Tensor): output, [batch_size, seq_len, d_model]
        """
        batch_size = Q.size(0)
        seq_len = Q.size(1)

        Q = self.W_Q(Q).view(batch_size, self.num_head, seq_len, self.d_h) # [batch_size, num_head, seq_len, d_k]
        K = self.W_K(K).view(batch_size, self.num_head, seq_len, self.d_h) # [batch_size, num_head, seq_len, d_k]
        V = self.W_V(V).view(batch_size, self.num_head, seq_len, self.d_h) # [batch_size, num_head, seq_len, d_k]

    
        seq_len = Q.shape[-2] # sequence length
        e = get_relative_positional_encoding(self.E, seq_len, self.d_h) # [num_head, seq_len, d_k]
        # calculate the Srel
        Srel = skew(torch.matmul(Q, e.transpose(-1, -2))) # [batch_size, num_head, seq_len, seq_len]
        
        # Calculate the attention scores
        # Relative Attention = Softmax((Q * K^T + Srel) / sqrt(d_h)) * V
        attention = relative_scaled_dot_product_attention(Q, K, V, Srel, mask)


        # concatenate the heads
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.fc(attention)
        return output
        