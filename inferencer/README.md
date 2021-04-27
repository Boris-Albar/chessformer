The Chess Inferencer

Plays games in rust using a policy/value model and mcts.

# Generate games

```
chessformer-inferencer --onnx_path ../trainer/checkpoints/ --games_path games_step0/ --parallel_games 8 --mcts_threads 4 --use_stochastic_moves --batch_size 32 --enable_cache --max_games 300 --mcts_moves 300 --use_time --policy_temperature 0.75
```
