"""
Compatibility shim - canonical implementation lives in ``src.nets.sac_policy_modul``.

Several existing SAC checkpoint ``.zip`` files were serialised with cloudpickle while
``ModulatedSACPolicy`` and friends were defined in ``src.utils.sac_policy_modul``.
cloudpickle stores the *module path* of each class at save-time; on load it does::

    module = importlib.import_module("src.utils.sac_policy_modul")
    cls    = getattr(module, "ModulatedSACPolicy")

Re-exporting all public names here means the lookup succeeds without modifying
any saved ``.zip`` files.
"""

# Re-export everything so that ``src.utils.sac_policy_modul.<Name>``
# resolves to the same object as ``src.nets.sac_policy_modul.<Name>``.
from src.nets.sac_policy_modul import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    ModulatedMlp,
    ModulatedActor,
    ModulatedContinuousCritic,
    ModulatedSACPolicy,
)

__all__ = [
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "ModulatedMlp",
    "ModulatedActor",
    "ModulatedContinuousCritic",
    "ModulatedSACPolicy",
]