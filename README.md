# Poker Agent Competition

OSU AI Club — Spring 2026

See `instructions.ipynb` for full competition details.

## Quick Start

```bash
pip install -r requirements.txt
```

1. Edit `agent.py` — implement your strategy in `MyAgent.act()`. If you are implementing a deep reinforcement learning agent, you may want to create additional files. See `instructions.ipynb` and the `rl_example` directory for more details.
2. Run `python test_agent.py` to benchmark against built-in opponents.

## Files

| File | Purpose |
|---|---|
| `agent.py` | **Your agent** — edit this |
| `test_agent.py` | Run competitions and print results |
| `poker_env.py` | Environment, hand evaluator, action/round enums |
| `agents.py` | RandomAgent, HeuristicAgent (opponents) |
| `rl_example/` | Optional deep RL reference (train, evaluate, deploy) |
| `instructions.ipynb` | Full competition guide |
