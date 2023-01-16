TicTacToe GPT Finetuning
========================

Simple python implementation of Tic Tac Toe.

Designed to make GPT able to recognize valid moves in Tic Tac Toe

::

   $ pip install tictactoe-gpt-finetuning

Examples
--------

Generate a game:

.. code:: python

    import tictactoe_gpt_finetuning as tictactoe
    print( tictactoe.generate_random_game() )

Generate many games:

.. code:: python

    import tictactoe_gpt_finetuning as tictactoe
    print( tictactoe.generate_n_games() )

Initialize and use the game board to place in top left:

.. code:: python

    import tictactoe_gpt_finetuning as tictactoe
    b = tictactoe.BoardState()
    b.make_move( 0, 0, 'x' )
    print( b )
    # output:
    # x - -
    # - - -
    # - - -
