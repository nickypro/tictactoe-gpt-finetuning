""" Fine-tune GPT-2 on Tic-Tac-Toe games.
"""
from typing import Optional
import time

import torch
from torch import Tensor
from transformers import pipeline #, set_seed

from tictactoe import tictactoe

class Model:
    """ A wrapper for the GPT-2 model to make it easier to use. """
    def __init__(self, model_name: str = 'gpt2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = model_name
        self.generator = pipeline('text-generation', model='gpt2')
        self.model = self.generator.model.to(self.device)
        self.tokenizer = self.generator.tokenizer
        self.transformer = self.model.transformer
        self.lm_head = self.model.lm_head
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def get_ids(self, text: str) -> Tensor:
        encoded_input = self.tokenizer(text, return_tensors='pt')
        return encoded_input['input_ids']

    def forward(self, input_ids: Tensor) -> Tensor:
        input_ids = input_ids.to(self.device)
        output = self.transformer(input_ids, output_hidden_states=False)
        return output.last_hidden_state

    def unembed(self, output_embeds):
        return self.lm_head(output_embeds.to(self.device))

    def get_logits(self, input_ids: Tensor) -> Tensor:
        output_embeds = self.forward(input_ids.to(self.device))
        logits = self.unembed(output_embeds)
        return logits

    def get_all_predictions(self,
            text:str = None,
            input_ids: Tensor = None
            ) -> Tensor:
        if input_ids is None:
            input_ids = self.get_ids(text)
        logits = self.get_logits(input_ids.to(self.device))
        return torch.argmax(logits, dim=-1)

    def detokenize(self,
            logits: Tensor = None,
            output_ids: Tensor = None,
            ) -> Tensor:
        if output_ids is None:
            output_ids = torch.argmax(logits, dim=-1)
        return self.tokenizer.decode(output_ids.cpu())

    def evaluate_ce_loss( self,
            text: Optional[str] = None,
            input_ids: Optional[Tensor] = None,
            logits: Optional[Tensor] = None
        ):
        """Cross entropy loss for predicting the next token

        Args:
            text (str, optional): The text to evaluat.
            input_ids (Tensor, optional): The input IDs from text to evaluate.
            logits (Tensor, optional): The pre-computed logits from text to evaluate.

        Returns:
            loss: Mean Cross-Entropy loss over tokens
        """
        if text is None and input_ids is None:
            raise ValueError( "Must provide either text or input_ids" )

        # Generate input token ids and output top k token ids
        if input_ids is None:
            input_ids = self.get_ids( text )
        if logits is None:
            logits = self.get_logits( input_ids.to(self.device) )

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        predicted_log_probs = log_probs[..., :-1, :].gather(
            dim=-1, index=input_ids[..., 1:, None]
        )[..., 0]
        return -predicted_log_probs.mean()

    def learn(self, texts):
        input_ids = self.get_ids(texts).to(self.device)
        loss = self.evaluate_ce_loss(input_ids=input_ids)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

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
