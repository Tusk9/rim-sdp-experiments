# Judge prompts

Templates for `eval_harness.py --judge-prompt-file` and `rejudge.py`. `{instruction}` and `{response}` are filled in by plain text replacement.

| File | Source | Output format |
|---|---|---|
| `qi2023_gpt4_judge.txt` | Qi et al. 2023, "Fine-tuning Aligned Language Models Compromises Safety", the HEx-PHI GPT-4 judge: `gpt-3.5/eval_utils/openai_policy_gpt4_judge.py` in [LLM-Tuning-Safety/LLMs-Finetuning-Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety) at commit `8a3b38f`, MIT License, Copyright (c) 2023 LLM-Tuning-Safety | `#thereason: ... #thescore: N` (text, not JSON) |

The text is copied verbatim. The only change is that the two `"%s"` slots became `"{instruction}"` and `"{response}"`. The original was written for `gpt-4-0613`, so any other judge model is a deviation and should be reported as one.

When a template contains `#thescore`, the harness calls the judge without JSON mode and reads the score from the `#thescore:` line.
