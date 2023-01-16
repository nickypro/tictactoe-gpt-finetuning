TicTacToe GPT Finetuning
========================

Simple python implementation of Tic Tac Toe.

Designed to make GPT able to recognize valid moves in Tic Tac Toe

# Examples
==========
- Generate a game:
```

import tictactoe*gpt*finetuning as tictactoe

print( tictactoe.generate*random*game() )

```

- generate many games:
```

import tictactoe*gpt*finetuning as tictactoe

print( tictactoe.generate*n*games() )

```

- Initialize and use the game board:
```

import tictactoe*gpt*finetuning as tictactoe

b = tictactoe.BoardState()

b.make_move( 0, 0, 'x' )

```
