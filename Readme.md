Rimworld-RL is a reinforcement learning project that aims to apply a novel unsupervised algorithm to simulate an environment similar to the popular game Rimworld and its mechanics. The project utilizes a neuromorphic architecture for reinforcement learning from real-valued observations, as described in the paper "A Neuromorphic Architecture for Reinforcement Learning from Real-Valued Observations" by Sérgio F. Chevtchenko et al.

Abstract
Reinforcement Learning (RL) provides a powerful framework for decision-making in complex environments. However, implementing RL in hardware-efficient and bio-inspired ways remains a challenge. This project introduces a novel Spiking Neural Network (SNN) architecture that addresses this challenge by solving RL problems with real-valued observations.

The proposed model incorporates multi-layered event-based clustering, along with Temporal Difference (TD)-error modulation and eligibility traces, to enhance its performance. Comparative experiments with a tabular actor-critic algorithm using eligibility traces and a state-of-the-art Proximal Policy Optimization (PPO) algorithm are conducted to evaluate the effectiveness of the proposed model. Results show that our network consistently outperforms the tabular approach and successfully discovers stable control policies in classic RL environments such as mountain car, cart-pole, and acrobot.

One of the key advantages of the proposed model is its appealing trade-off between computational complexity and hardware implementation requirements. Unlike traditional RL methods, the model does not rely on an external memory buffer or global error gradient computation. Instead, synaptic updates occur online, driven by local learning rules and a broadcasted TD-error signal. This characteristic makes the proposed model more hardware-efficient and paves the way for the development of practical RL solutions.