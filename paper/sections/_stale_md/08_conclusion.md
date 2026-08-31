8. Conclusion

Tool-call interfaces censor agent trajectories, and the censoring is silent, family-specific,
scale-increasing, and — in reinforcement learning — capable of making an entire policy
branch unreachable by gradient. Repairing the interface restores the mechanism but recovers
only part of the outcome gap, and a genuine protocol difference remains underneath.

The most uncomfortable finding is methodological. We ran three single-variable paired
controls on a 23% failure and concluded it was a model deficiency; a fourth control, taken
from a sentence in the serving documentation, reduced it to zero. **The mechanism this
paper describes claimed us as one of its instances**, which is the strongest argument we can
offer that reporting server-parsed tool-call rates as model capability is not a safe default.

---
