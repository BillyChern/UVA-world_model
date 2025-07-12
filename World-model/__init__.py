# Automatically import main public classes
def __getattr__(name):
    if name == "UVAWorldModelEnv":
        from .uva_world_env import UVAWorldModelEnv
        return UVAWorldModelEnv
    raise AttributeError(name)

__all__ = ["UVAWorldModelEnv"] 