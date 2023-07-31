# import torch


# a = torch.tensor([[1,2,3,4,0,0,0,0],[1,2,3,5,6,4,0,0],[1,2,4,0,0,0,0,0],[1,2,3,6,8,9,10,4]])
# _, index = torch.where(a == 4)
# bs, seq_len = a.shape
# mask_val = torch.arange(seq_len).unsqueeze(0).repeat(bs, 1)
# mask_val = mask_val[mask_val[:,:] != index.reshape(-1,1)].reshape(-1, seq_len - 1)
# final_mask = torch.gather(a, 1, mask_val)
# print(final_mask)

# # print(index)







# multi head attention test

# import torch
# import torch.nn as nn 

# multi_head_attention = nn.MultiheadAttention(embed_dim = 256, num_heads = 8, dropout = 0.1, batch_first=True)

# L = 34
# S = 24

# query = torch.rand(2,L,256)
# key = torch.rand(2,S,256)

# attn_mask = torch.tril(torch.ones(L,S))
# key_padding_mask = torch.cat([torch.ones(2, L-2), torch.zeros(2,2)],dim=1)

# attn_matrix, attention_weights = multi_head_attention(
#     query, 
#     key,
#     key,
#     attn_mask = attn_mask,
#     key_padding_mask = key_padding_mask
# )

# print(attn_matrix.shape)
# print(attention_weights.shape)