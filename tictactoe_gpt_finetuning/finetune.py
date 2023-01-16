""" Fine-tune GPT-2 on Tic-Tac-Toe games.
"""
import time
import torch

# Package imports depends on if this is run as a script or imported
if __name__ == "__main__":
    # pylint: disable=import-error
    import tictactoe
    from model import Model

else:
    from . import tictactoe
    from .model import Model

def compare_tictactoe_predictions(gpt: Model):
    input_text = tictactoe.generate_random_game()
    input_ids = gpt.get_ids(input_text)
    output_ids = gpt.get_all_predictions(input_ids=input_ids)

    in_ids = input_ids.squeeze().detach().cpu()
    out_ids = output_ids.squeeze().detach().cpu()

    left_ids, right_ids = [ in_ids[0] ], [ in_ids[0] ]
    num_inputs = input_ids.size(1)
    for i in range(1, num_inputs):
        left_ids.append(in_ids[i])
        right_ids.append(out_ids[i-1])

    left_ids = torch.stack(left_ids)
    right_ids = torch.stack(right_ids)

    # Print comparision
    def sanitize(word):
        if word == '\n':
            return '\\n', 1
        if word == '\n\n':
            return '\\n\\n', 2
        return word, 0

    left_str, right_str = "", ""
    for i in range(num_inputs):
        left_word = gpt.tokenizer.decode(left_ids[i].item())
        right_word = gpt.tokenizer.decode(right_ids[i].item())

        left_word, newline = sanitize(left_word)
        right_word, _      = sanitize(right_word)

        if newline:
            left_word = ""
            right_word = ""

        left_str   = left_str  + left_word
        right_str  = right_str + right_word

        for _n in range(newline):
            print( "|%8s|%8s|" % (left_str, right_str) )
            left_str, right_str = "", ""

    print( "|%8s|%8s|" % (left_str, right_str) )
    return

def finetune(gpt: Model, n_epochs: int = 1, batch_size: int = 20):
    for epoch in range(n_epochs):
        texts = tictactoe.generate_n_games(batch_size)
        loss = gpt.learn(texts)
        print( f"Epoch {epoch} Loss: {loss.item()}" )

    return

if __name__ == "__main__":
    _gpt = Model()
    # Look at what outputs look like initially
    compare_tictactoe_predictions(_gpt)

    # Fine tune the model
    t0 = time.time()
    finetune(_gpt, n_epochs=100, batch_size=16)
    print( time.time() - t0 )

    # Look at what outputs look like after training
    compare_tictactoe_predictions(_gpt)
