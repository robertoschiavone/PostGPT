import torch

from BigramLanguageModel import BigramLanguageModel


# take a string, output a list of integers
def encode(s, stoi):
    return [stoi[c] for c in s]


# take a list of integers, output a string
def decode(l, itos):
    return "".join([itos[i] for i in l])


@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size,
                  device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, block_size, batch_size,
                             device)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


# generate a small batch of data of inputs x and targets y
def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


if __name__ == "__main__":
    # hyperparameters
    batch_size = 32  # how many independent sequence will be processed in parallel
    block_size = 8  # what is the maximum context length for predictions
    max_steps = 5_000
    eval_interval = 500
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_iters = 200
    n_embed = 32

    torch.manual_seed(1337)

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # train and test splits
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = BigramLanguageModel(vocab_size, n_embed, block_size, device)
    model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # created once at th beginning of training
    scaler = torch.cuda.amp.GradScaler()

    for step in range(max_steps):
        # every once in a while evaluate the loss on train and val sets
        if step % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, train_data, val_data,
                                   block_size, batch_size, device)
            print(f"Step {step}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch("train", train_data, val_data, block_size, batch_size,
                           device)

        optimizer.zero_grad(set_to_none=True)

        # cast operations to mixed precision
        with torch.cuda.amp.autocast():
            _, loss = model(xb, yb)

        # scale the loss and create scaled gradients
        scaler.scale(loss).backward()

        # unscale gradients
        scaler.step(optimizer)

        # update scaler for next iteration
        scaler.update()

    losses = estimate_loss(model, eval_iters, train_data, val_data, block_size,
                           batch_size, device)
    print(f"Step {max_steps}: train loss {losses['train']:.4f}, "
          f"val loss {losses['val']:.4f}")

    # generate from model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist(), itos))
