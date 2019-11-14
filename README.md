# PhD
Work In PhD
## Code from other repo as reference (to be copied from lol)
## FIRMA
### Data
removed one subject (11013) from dataset because it is too small, and now we have 17 subject left.
### GAN_LSTM
#### lstm_autoencoder
In the decoder part, we are only decoding one token at a time, the input tokens will always have a sequence length of 1. We can thus easily  control the teacher forcing. In Pytorch, if the input always has sequence length of 1, it's natural to think of using LSTMcell/GRUcell instead of LSTM/GRU, but that would make it harder to stack up layers, so usually we would just use LSTM/GRU and use the "unsqueeze" function to process the given input to give it an extra dimension for sequence length. And because decoding process is done one by one and the encoding process is not, usually the encoder and decoder are defined separately. 
Reference: https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
