import torch


# take a string, output a list of integers
def encode(s, stoi):
    return [stoi[c] for c in s]


# take a list of integers, output a string
def decode(l, itos):
    return "".join([itos[i] for i in l])


# generate a small batch of data of inputs x and targets y
def get_batch(split, train_data, val_data, block_size, batch_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y


if __name__ == "__main__":
    # read it in to inspect it
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Length of dataset in characters: {len(text)}")

    # look at the first 1000 characters
    print(text[:1000])

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("".join(chars))
    print(vocab_size)

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(encode("Hello world!", stoi))
    print(decode(encode("Hello world!", stoi), itos))

    # encode the entire text dataset and store it into a torch.Tensor
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    print(data.shape, data.dtype)
    # the 1000 characters we looked at earlier will look to the GPT
    # like this
    print(data[:1000])

    # split up the data into train and validation sets
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    block_size = 8
    print(train_data[:block_size + 1])

    x = train_data[:block_size]
    y = train_data[1:block_size + 1]
    for t in range(block_size):
        context = x[:t + 1]
        target = y[t]
        print(f"When input is {context}, target is {target}")

    torch.manual_seed(1337)
    batch_size = 4  # how many independent sequences will we process in parallel?
    block_size = 8  # what is the maximum context length for predictions?

    xb, yb = get_batch("train", train_data, val_data, block_size, batch_size)
    print(f"Inputs:\n{xb.shape}\n{xb}\n")
    print(f"Targets:\n{yb.shape}\n{yb}\n")

    print("-----")

    for b in range(batch_size):  # batch dimension
        for t in range(block_size):  # time dimension
            context = xb[b, :t + 1]
            target = yb[b, t]
            print(f"When input is {context}, target is {target}")
