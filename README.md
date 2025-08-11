# Aim-and-Shoot

## Modeling Visually-Guided Aim-and-Shoot Behavior in First-Person Shooters (IJHCS'25)

- ### Simulated FPS player with human-like perceptual and motor skills
  - We created a computationally rational agent with human-like perception and motor abilities, trained via reinforcement learning.
  - The model parameters include motor noise, position perception noise, speed perception noise, internal clock precision, and four motivational states.
  - The simulation agent adapts to wide range of above 8 model parameters (i.e., user features).
  - Our simulated agent replicated the trial completion time, accuracy, and saccadic deviation of human players (N=20).
- ### [Paper] (https://doi.org/10.1016/j.ijhcs.2025.103503)

## Datasets
The dataset includes behavaiors of 10 professional and 10 amateur FPS players.
The trajectory data is not included in github due to its size.
If you need the full data, please contact via jsyoon.k5 at gmail.com.


## Citation

- Please cite this paper as follows if you use this code in your research.

```
@article{yoon2025modeling,
  title={Modeling visually-guided aim-and-shoot behavior in first-person shooters},
  author={Yoon, June-Seop and Moon, Hee-Seung and Boudaoud, Ben and Spjut, Josef and Frosio, Iuri and Lee, Byungjoo and Kim, Joohwan},
  journal={International Journal of Human-Computer Studies},
  volume={199},
  pages={103503},
  year={2025},
  publisher={Elsevier}
}
```
