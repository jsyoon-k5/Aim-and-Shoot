### Reference:
### https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/policies.py
### https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py

### Code modified by Hee-Seung Moon

import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import make_proba_distribution


# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class ModulatedMlp(nn.Module):
    """
    Mlp networks which can be moduluated by concatenation (+ embedding network)
    -----
    :param net_arch: Network architecture
    :param features_dim: Number of features
    :param sim_param_dim: Number of simulation paramters
    :param output_dim: if output_dim > 0 (default = -1), there will be additional linear layer without activation at the last
    :param activation_fn: Activation function
    :param embed_net_arch: The specification of the embedding networks (if use)
    :param concat_layers: Which layer to be concatenated with embedding params (eg. [0, 2] --> input, 2nd hidden layers)
    """
    def __init__(
        self,
        net_arch: List[int],
        features_dim: int,
        sim_param_dim: int,
        output_dim: int = -1,
        activation_fn: Type[nn.Module] = nn.ReLU,
        embed_net_arch: Optional[List[int]] = None,
        concat_layers: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
    ):
        super().__init__()
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.sim_param_dim = sim_param_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn

        self.concat_layers = concat_layers
        if embed_net_arch is None:
            self.embed_dim = self.sim_param_dim
        else:
            self.embed_dim = embed_net_arch[-1]

        self.layers = nn.ModuleList([])
        self.last_layer_dim = features_dim - self.sim_param_dim
        for i in range(len(net_arch)):
            n_in = self.last_layer_dim
            n_in += self.embed_dim if i in self.concat_layers else 0
            self.last_layer_dim = net_arch[i]
            self.layers.append(nn.Linear(n_in, self.last_layer_dim))

        if len(net_arch) in self.concat_layers:
            self.last_layer_dim += self.embed_dim
        if output_dim > 0:
            self.layers.append(nn.Linear(self.last_layer_dim, output_dim))
            self.last_layer_dim = output_dim
        
        self.embed_nets = nn.ModuleList([])
        for _ in range(len(self.concat_layers)):
            embed_net = create_mlp(
                self.sim_param_dim,
                self.embed_dim,
                embed_net_arch[:-1],
                self.activation_fn
            ) if embed_net_arch is not None else []
            self.embed_nets.append(nn.Sequential(*embed_net))

    def forward(self, input: th.Tensor) -> th.Tensor:
        sim_param, feat_param = input.to(th.float32).split(
            [self.sim_param_dim, self.features_dim - self.sim_param_dim],
            -1,
        )
        x = feat_param
        for i in range(len(self.net_arch)):
            if i in self.concat_layers:
                embed = self.embed_nets[self.concat_layers.index(i)](sim_param)
                x = th.cat([x, embed], dim=-1)
            x = self.layers[i](x)
            x = self.activation_fn()(x)

        if len(self.net_arch) in self.concat_layers:
            embed = self.embed_nets[self.concat_layers.index(len(self.net_arch))](sim_param)
            x = th.cat([x, embed], dim=-1)
        if self.output_dim > 0:
            x = self.layers[-1](x)
        return x


class ModulatedActor(Actor):
    """
    Actor class (described below) enabling being conditioned on simulation parameters
    ------
    Actor network (policy) for SAC.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    ------
    :param sim_param_dim: Dimension of simulation paramters
    :param embed_net_arch: The specification of the embedding networks (if use)
    :param concat_layers: Which layer to be concatenated with embedding params (eg. [0, 2] --> input, 2nd hidden layers)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        sim_param_dim: int, ### <-- Added argument
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        embed_net_arch: Optional[List[int]] = None, ### <-- Added argument
        concat_layers: Optional[List[Union[int, Dict[str, List[int]]]]] = None, ### <-- Added argument
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        action_dim = get_action_dim(self.action_space)

        ###### Modification made
        self.latent_pi_net = ModulatedMlp(
            net_arch,
            features_dim,
            sim_param_dim,
            output_dim=-1,
            activation_fn=activation_fn,
            embed_net_arch=embed_net_arch,
            concat_layers=concat_layers
        )
        last_layer_dim = self.latent_pi_net.last_layer_dim
        self.current_log_std = None
        ###### Until here

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)
        

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.
        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.features_extractor(obs)
        ###### Modification made
        latent_pi = self.latent_pi_net(features)
        ###### Until here
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.current_log_std = log_std.detach()

        return mean_actions, log_std, {}


class ModulatedContinuousCritic(ContinuousCritic):
    """
    ContinuousCritic class (described below) enabling being conditioned on simulation parameters
    ------
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.
    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    ------
    :param sim_param_dim: Dimension of simulation paramters
    :param embed_net_arch: The specification of the embedding networks (if use)
    :param concat_layers: Which layer to be concatenated with embedding params (eg. [0, 2] --> input, 2nd hidden layers)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        sim_param_dim: int, ### <-- Added argument
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        embed_net_arch: Optional[List[int]] = None, ### <-- Added argument
        concat_layers: Optional[List[Union[int, Dict[str, List[int]]]]] = None, ### <-- Added argument
    ):
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            ###### Modification made
            q_net = ModulatedMlp(
                net_arch,
                features_dim + action_dim,
                sim_param_dim,
                output_dim=1,
                activation_fn=activation_fn,
                embed_net_arch=embed_net_arch,
                concat_layers=concat_layers
            )
            ###### Until here
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class ModulatedSACPolicy(SACPolicy):
    """
    SACPolicy class (described below) enabling being conditioned on simulation parameters
    ------
    Policy class (with both actor and critic) for SAC.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    ------
    :param sim_param_dim: Dimension of simulation paramters
    :param embed_net_arch: The specification of the embedding networks (if use)
    :param concat_layers: Which layer to be concatenated with embedding params (eg. [0, 2] --> input, 2nd hidden layers)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        sim_param_dim: int, ### <-- Added argument
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        embed_net_arch: Optional[List[int]] = None, ### <-- Added argument
        concat_layers: Optional[List[Union[int, Dict[str, List[int]]]]] = None, ### <-- Added argument
    ):
        super(SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        ###### Modification made
        self.actor_kwargs.update(
            {
                "sim_param_dim": sim_param_dim,
                "embed_net_arch": embed_net_arch,
                "concat_layers": concat_layers,
            }
        )
        self.critic_kwargs.update(
            {
                "sim_param_dim": sim_param_dim,
                "embed_net_arch": embed_net_arch,
                "concat_layers": concat_layers,
            }
        )
        ###### Until here

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    ###### Modification made
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ModulatedActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ModulatedActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ModulatedContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ModulatedContinuousCritic(**critic_kwargs).to(self.device)
    ###### Until here