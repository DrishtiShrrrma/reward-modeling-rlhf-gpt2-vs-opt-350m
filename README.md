# reward-modeling-rlhf-gpt2-vs-opt-350m


This study focuses on comparing two renowned models: Facebook’s opt-350m and GPT-2. Both models are trained and tested on the Anthropic/hh-rlhf dataset having 160800 rows, offering insights into their respective performances and potential strengths.

## Data Preprocessing and Reward Modeling
The foundation of any ML experiment is accurate data processing. The Anthropic/hh-rlhf dataset comprises paired samples of "chosen" and "rejected" sequences. We will define a preprocessing function which will tokenize, truncate and format the sequences to maintain a uniform structure.

TRL supports custom reward modeling to perform reward modeling on dataset and model. The reward trainer expects a very specific format for the dataset. Since the model will be trained to predict which sentence is the most relevant, given two sentences.

The entries should be named:

- input_ids_chosen
- attention_mask_chosen
- input_ids_rejected
- attention_mask_rejected

## Setting Up Reward-based Training
The aim was to fine-tune the models using rewards. This is where the trl library comes into play. We used the RewardTrainer class:

![image](https://github.com/DrishtiShrrrma/reward-modeling-rlhf-gpt2-vs-opt-350m/assets/129742046/7562f5de-1ff3-4404-8f4d-33a57c315471)

## Insights & Results

![image](https://github.com/DrishtiShrrrma/reward-modeling-rlhf-gpt2-vs-opt-350m/assets/129742046/1b95feca-6a41-4af5-ab31-b04ecd6ff22e)

1. **Training Loss:** From the results, it’s evident that both models converge towards a similar loss by the 10,000th step. However, the trajectory of the decline in loss differs. opt-350m starts with a higher initial loss but achieves a steeper descent, while GPT-2, despite starting with a steeper loss, reduces it at a more gradual pace. This suggests that the OPT-350M might have an initial advantage in terms of understanding or adapting to the dataset, but both models eventually reach a similar level of performance.

**2. Training Time:** GPT-2 demonstrated a quicker training speed, completing in 33:18, while opt-350m required 1:23:54. The difference in training duration might be attributed to the inherent architecture of each model. GPT-2’s design may be inherently more optimized for tasks resembling those in the Anthropic/hh-rlhf dataset. Alternatively, GPT-2 might be better suited or more compatible with the specific characteristics of the Anthropic/hh-rlhf dataset, leading to quicker convergence.

## Future Exploration
It would be worthwhile to delve deeper into why such a significant difference in training time exists, considering both models are prominent in the NLP field. This could lead to insights about model architecture, optimization, or even dataset-specific peculiarities.

## Conclusion
In our study comparing opt-350m and GPT-2 on the Anthropic/hh-rlhf dataset, distinct training patterns emerged. While opt-350m experienced a rapid initial decline in loss, GPT-2 showed a steadier descent but trained faster overall. This difference underscores the unique efficiencies of each model and their interaction with the dataset. It’s essential to recognize that our observations depend on various factors, including dataset choice and preprocessing. Reward-based training emerges as a potent technique in language modeling, highlighting the need to match model strengths to specific tasks for optimal outcomes.







