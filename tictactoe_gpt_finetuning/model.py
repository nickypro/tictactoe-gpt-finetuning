from typing import Optional

import torch
from torch import Tensor
from transformers import pipeline #, set_seed

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
