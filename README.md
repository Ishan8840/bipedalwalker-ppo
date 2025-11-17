# 🏃‍♂️ BipedalWalker PPO with Curriculum Learning

This project implements **Proximal Policy Optimization (PPO)** to train an agent for **BipedalWalker-v3** using **automatic curriculum learning**, progressing from easy to hardcore terrain.

---

## 📖 Theoretical Foundations

This implementation is based on the **Proximal Policy Optimization (PPO)** algorithm from the paper:

> Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. ([arXiv][1])

Key points from the paper:

* PPO alternates between **sampling data** via environment interaction and **optimizing a surrogate objective** over multiple epochs. ([arXiv][1])
* It uses a **clipped probability ratio objective**, which prevents large updates to the policy and stabilizes training. ([jsDelivr][2])
* PPO achieves a balance between sample efficiency, simplicity, and computational cost compared to more complex algorithms like TRPO. ([transferlab.ai][3])

In this repository, your implementation of PPO follows these principles:

1. **Clipped objective**: you compute the ratio of new policy to old policy log-probabilities, and clip according to `eps_clip`.
2. **Multiple epochs / minibatch updates**: instead of a single gradient step per trajectory, you run several epochs of updates over minibatches.
3. **Advantage estimation**: you use GAE (Generalized Advantage Estimation), which is commonly paired with PPO for more stable advantage computation.

---

## 🚀 Features (Updated)

* **PPO-based Training**: Using the clipped surrogate objective as described in the original PPO paper.
* **Curriculum Learning**: Automatic transition from flat terrain to hardcore mode once performance threshold is met.
* **Actor-Critic Architecture**: Separate policy (actor) and value (critic) networks, trained via PPO.
* **Stable Updates**: Implements multi-epoch minibatch updates and normalized GAE.

---

## 📚 Usage

### Installation

```bash
pip install requirements.txt
```

### Training

```bash
python ppo.py
```

* The script runs PPO with curriculum learning.
* It saves model checkpoints when leveling up:

  * `actor_level_0.pth`
  * `critic_level_0.pth`
  * `actor_level_1.pth`
  * etc.

### Evaluation / Rendering

After training, run evaluation in hardcore mode:

```python
env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
```

Then sample actions from the trained policy, render, and observe.

---

## 🔧 Hyperparameters

Here are some of the key hyperparameters used in your implementation:

| Parameter                      | Value |
| ------------------------------ | ----- |
| `n_steps`                      | 1,600 |
| `batch_size`                   | 64    |
| `n_epochs`                     | 10    |
| `gamma`                        | 0.99  |
| `lambda` (GAE)                 | 0.95  |
| `eps_clip`                     | 0.2   |
| `actor LR`                     | 1e-4  |
| `critic LR`                    | 1e-3  |
| `reward threshold to level up` | ~300  |

These are quite similar in spirit to settings used in PPO research, calibrated for BipedalWalker.

---

## 📂 Project Structure (Suggested)

```
bipedalwalker-ppo-curriculum/
│
├── train.py                # main training script
├── actor_level_*.pth        # saved actor checkpoints
├── critic_level_*.pth       # saved critic checkpoints
├── requirements.txt         # dependencies
└── README.md                # this documentation
```

---

## 📈 Expected Performance

* You should expect to **solve flat terrain** first, achieving high rewards there.
* After leveling up, the policy should **transfer to hardcore environment**, leveraging what was learned in the simpler environment.
* Curriculum learning speeds up convergence and stabilizes training compared to naive PPO on the hardest terrain from scratch.

---

## 🙏 Acknowledgments & References

* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. ([arXiv][1])
* OpenAI Gym / Gymnasium
* Inspired by **OpenAI Spinning Up** design principles
