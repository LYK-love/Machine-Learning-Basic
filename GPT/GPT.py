import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout_prob, block_size):
        super().__init__()
        self.head_size, self.n_embd, self.dropout_prob, self.block_size = head_size, n_embd, dropout_prob, block_size
        
        self.key = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(self.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(self.n_embd, self.head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        # x.shape = [B, T, head_size]
        B,T,C = x.shape # Here, `C` = `head_size` instead of num of embeddings.
        k = self.key(x) # [B, T, C]
        q = self.query(x) # [B, T, C]
        
        # Self attention
        # For the last dimention, each row represents the "attention" of the corresponding token to all the tokens in the context. 
        wei = q @ k.transpose(-2, -1) # [B, T, T]
        # Now masking out future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # [B, T, T]
        
        wei = F.softmax(wei, dim=-1) # [B, T, T]. 
        wei = self.dropout(wei) # WHY?
        
        v = self.value(x) # (B,T, C)
        out = wei @ v # [B, T, C]
        return out
        

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size,n_embd, dropout_prob, block_size):
        super().__init__()
        self.num_heads, self.head_size, self.n_embd, self.dropout_prob, self.block_size = num_heads, head_size, n_embd, dropout_prob, block_size
        
        self.heads = nn.ModuleList([Head(self.head_size, self.n_embd, self.dropout_prob, self.block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(self.n_embd, self.n_embd) # Mix the data output by the heads back together.
        self.dropout = nn.Dropout(self.dropout_prob)
        

    def forward(self, x):
        outs = torch.cat([h(x) for h in self.heads], dim=-1) # [B, T, C]. Each `out` is [B, T, head_size]
        x = self.dropout(self.proj(outs)) # [B, T, C]
        return x
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        
        # very simple tranformation
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ 
    Transformer block: communication followed by computation 
    Does not include the cross-attention layer sicne it's a decoder-only transformer.
    """

    def __init__(self, n_embd, n_head, dropout_prob, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.n_embd, self.n_head, self.dropout_prob, self.block_size  = n_embd, n_head, dropout_prob, block_size
        self.head_size = self.n_embd // self.n_head # The total effect will be equivalen to use a single head.
        
        self.sa = MultiHeadAttention(self.n_head, self.head_size, self.n_embd, self.dropout_prob, self.block_size)
        self.ffwd = FeedFoward(self.n_embd) # [B, T, C] -> [B, T, C]
        self.ln_1 = nn.LayerNorm(self.n_embd) # [B, T, C] -> [B, T, C]
        self.ln_2 = nn.LayerNorm(self.n_embd) # [B, T, C] -> [B, T, C]
    def forward(self, x):
        '''
        @x: [B, T, C], the embedding of the input tokens.
        '''
        B, T, C = x.shape
        
        # Residual connection
        x = x + self.sa(self.ln_1(x)) # [B, T, C] # Why applying ln before the self-attention?
        x = x + self.ffwd(self.ln_2(x)) # [B, T, C] # Why applying ln before the feed forward layer?
        return x
    

    
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout_prob, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #  [V, C] # The token idx ranges from 0 to V-1.
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # [T, C] # The position ranges from 0 to T-1.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout_prob, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # [C, V] # The output logits for the next token.

        self.vocab_size, self.n_embd, self.block_size, self.n_head, self.n_layer, self.dropout_prob, self.device = vocab_size, n_embd, block_size, n_head, n_layer, dropout_prob, device
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        
        tok_emb = self.token_embedding_table(idx) # [B, T, C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # [T, C]
        x = tok_emb + pos_emb # [B, T, C] # WHY???
        x = self.blocks(x) # [B, T, C]
        x = self.ln_f(x) # [B, T, C] # In paper this doesn't happen. Why?
        logits = self.lm_head(x) # [B, T, V]
        
        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] # only use the last `block_size` tokens as the condition to generate.
            
            # Now `idx_cond.shape` = [B, T] where T is the context length.
            B, T = idx_cond.shape
            # forward pass, get the logits for next token
            logits, _ = self(idx_cond) # [B, T, V]. 
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # [B, V]. Only predict the next token for the last token in the context. Apply softmax to the class scores (dim=-1).
            idx_next = torch.multinomial(probs, num_samples=1) # [B, 1]
            idx = torch.cat((idx, idx_next), dim=1) # [B, T+1]
        return idx # [B, T+`max_new_tokens`]
    

if __name__ == "__main__":
    

    # hyperparameters
    batch_size = 16 # how many independent sequences will we process in parallel?
    block_size = 32 # what is the maximum context length for predictions?
    max_iters = 25000
    eval_interval = 100
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0
    # ------------

    torch.manual_seed(1337)

    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('./input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device=device)
    m = model.to(device)
    # print the number of parameters in the model
    # For each parameter tensor p, the .numel() method returns the total number of elements contained in the tensor. 
    # This is effectively the product of the tensor's shape dimensions. For instance, if a weight tensor p has a shape of [3, 5], it means the tensor has 15 elements (3 * 5 = 15).
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') 

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

