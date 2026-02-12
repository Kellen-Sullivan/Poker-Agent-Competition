# Poker-Agent-Competition

Look at instructions.ipynb to learn more about the competition and how to participate!

## Project Structure

```
├── texas_holdem_env/     # Texas Hold'em environment (Action, Round, TexasHoldEm)
├── examples/             # poker_eval, example_usage, example_agents
├── workspace/            # Your code goes here
│   ├── agent.py          # Implement your agent
│   └── train.py          # Implement your training loop
└── test_agent.py         # Run competitions and benchmark
```

## Quick Start

1. Implement your agent in `workspace/agent.py` (must have `act(observation, env)` method).
2. Run `python test_agent.py` from the project root to benchmark.
