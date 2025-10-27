"""Game environment wrappers"""

from .chrome_dino_env import ChromeDinoEnv, SimpleDinoEnv

try:
    from .vizdoom_env import (ViZDoomEnv, ViZDoomEnvWithPaperReward,
                              create_vizdoom_env)

    VIZDOOM_AVAILABLE = True
except ImportError:
    VIZDOOM_AVAILABLE = False
    print("Warning: ViZDoom not available. Install with: pip install vizdoom")
