from stable_baselines3.common.callbacks import BaseCallback
import torch as th

class TensorboardStdCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardStdCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        actor = self.model.policy.actor

        # Ensure that current_log_std is a tensor
        if actor.current_log_std is not None:
            std = th.exp(actor.current_log_std).mean().item()
            self.logger.record('rollout/action_std', std)
        else:
            self.logger.record('rollout/action_std', float('nan'))

        if "infos" in self.locals and self.locals["infos"]:
            info = self.locals["infos"][-1]
            

        return True
