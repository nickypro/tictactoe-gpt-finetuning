# TicTacToe GPT Finetuning

Simple python implementation of Tic Tac Toe.

Designed to make GPT able to recognize valid moves in Tic Tac Toe

```
$ pip install welford-torch
```

## Examples

Generate a game:

```
import tictactoe_gpt_finetuning as tictactoe
print( tictactoe.generate_random_game() )
```

Generate many games:

```
import tictactoe_gpt_finetuning as tictactoe
print( tictactoe.generate_n_games() )
```

Initialize and use the game board to place in top left:

```
import tictactoe_gpt_finetuning as tictactoe
b = tictactoe.BoardState()
b.make_move( 0, 0, 'x' )
print( b )
```

Output:
```
 x - -
 - - -
 - - -
```