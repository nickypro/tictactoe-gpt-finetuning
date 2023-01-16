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

    from tictactoe_gpt_finetuning import tictactoe
    print( tictactoe.generate_random_game() )

Generate many games:

.. code:: python

    from tictactoe_gpt_finetuning import tictactoe
    print( tictactoe.generate_n_games() )

Initialize and use the game board to place in top left:

.. code:: python

    from tictactoe_gpt_finetuning import tictactoe
    b = tictactoe.BoardState()
    b.make_move( 0, 0, 'x' )
    print( b )
    # output:
    # x - -
    # - - -
    # - - -

Train a Model
-------------

We can compare inputs to outputs of the model, and compare
predictions of the model before and after finetuning.

.. code:: python

    from tictactoe import Model, finetune, compare_tictactoe_predictions
    gpt = Model()

    # See what predictions look like before finetuning
    compare_tictactoe_predictions( gpt )

    # Fine-tune the model
    finetune( gpt, n_epochs=10 )

    # See what new predictions look like after finetuning
    compare_tictactoe_predictions

