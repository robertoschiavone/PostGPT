import torch


# take a string, output a list of integers
def encode(s, stoi):
    return [stoi[c] for c in s]


# take a list of integers, output a string
def decode(l, itos):
    return "".join([itos[i] for i in l])


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
