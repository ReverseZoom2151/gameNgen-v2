"""
ViZDoom Environment Wrapper for GameNGen
Gymnasium-compatible wrapper for DOOM game
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional
import vizdoom as vzd
from pathlib import Path


class ViZDoomEnv(gym.Env):
    """
    ViZDoom Environment for GameNGen

    Observation Space: RGB image (H, W, 3)
    Action Space: Discrete(43) - DOOM's action space
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    # DOOM action space (43 actions total)
    # Based on combinations of: move forward/back, turn left/right, shoot, etc.
    ACTIONS = [
        # Format: [MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT,
        #          TURN_LEFT, TURN_RIGHT, ATTACK, SPEED, USE]
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: No action
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Move forward
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2: Move backward
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3: Move left
        [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 4: Move right
        [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5: Turn left
        [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6: Turn right
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 7: Attack
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 8: Speed
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 9: Use
        # Combinations
        [1, 0, 0, 0, 1, 0, 0, 0, 0],  # 10: Forward + Turn left
        [1, 0, 0, 0, 0, 1, 0, 0, 0],  # 11: Forward + Turn right
        [1, 0, 0, 0, 0, 0, 1, 0, 0],  # 12: Forward + Attack
        [0, 0, 0, 0, 1, 0, 1, 0, 0],  # 13: Turn left + Attack
        [0, 0, 0, 0, 0, 1, 1, 0, 0],  # 14: Turn right + Attack
        [1, 0, 0, 0, 1, 0, 1, 0, 0],  # 15: Forward + Turn left + Attack
        [1, 0, 0, 0, 0, 1, 1, 0, 0],  # 16: Forward + Turn right + Attack
        [0, 1, 0, 0, 1, 0, 0, 0, 0],  # 17: Backward + Turn left
        [0, 1, 0, 0, 0, 1, 0, 0, 0],  # 18: Backward + Turn right
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 19: Strafe left
        [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 20: Strafe right
        [1, 0, 1, 0, 0, 0, 0, 0, 0],  # 21: Forward + Strafe left
        [1, 0, 0, 1, 0, 0, 0, 0, 0],  # 22: Forward + Strafe right
        [0, 1, 1, 0, 0, 0, 0, 0, 0],  # 23: Backward + Strafe left
        [0, 1, 0, 1, 0, 0, 0, 0, 0],  # 24: Backward + Strafe right
        [1, 0, 1, 0, 0, 0, 1, 0, 0],  # 25: Forward + Strafe left + Attack
        [1, 0, 0, 1, 0, 0, 1, 0, 0],  # 26: Forward + Strafe right + Attack
        [0, 0, 1, 0, 0, 0, 1, 0, 0],  # 27: Strafe left + Attack
        [0, 0, 0, 1, 0, 0, 1, 0, 0],  # 28: Strafe right + Attack
        [1, 0, 0, 0, 0, 0, 0, 1, 0],  # 29: Forward + Speed
        [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 30: Turn left + Speed
        [0, 0, 0, 0, 0, 1, 0, 1, 0],  # 31: Turn right + Speed
        [1, 0, 0, 0, 1, 0, 0, 1, 0],  # 32: Forward + Turn left + Speed
        [1, 0, 0, 0, 0, 1, 0, 1, 0],  # 33: Forward + Turn right + Speed
        [1, 0, 0, 0, 0, 0, 1, 1, 0],  # 34: Forward + Attack + Speed
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 35: Idle (duplicate, for symmetry)
        [1, 0, 0, 0, 1, 0, 1, 1, 0],  # 36: Forward + Turn left + Attack + Speed
        [1, 0, 0, 0, 0, 1, 1, 1, 0],  # 37: Forward + Turn right + Attack + Speed
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 38: Reserved
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 39: Reserved
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 40: Reserved
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 41: Reserved
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 42: Reserved
    ]

    def __init__(
        self,
        config_file: str = "basic.cfg",
        width: int = 320,
        height: int = 256,
        frame_skip: int = 4,
        render_mode: Optional[str] = None,
        visible: bool = False,
        scenarios_dir: Optional[str] = None,
    ):
        """
        Args:
            config_file: ViZDoom config file (e.g., 'basic.cfg')
            width: Frame width
            height: Frame height
            frame_skip: Apply each action for N frames (paper uses 4)
            render_mode: 'rgb_array' or 'human'
            visible: Whether to show game window
            scenarios_dir: Directory containing scenario files
        """
        super().__init__()

        self.width = width
        self.height = height
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.config_file = config_file
        self.scenarios_dir = scenarios_dir

        # Action space: 43 discrete actions
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # Observation space: RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

        # Initialize ViZDoom game
        self.game = vzd.DoomGame()

        # Load configuration
        if scenarios_dir:
            config_path = Path(scenarios_dir) / config_file
        else:
            # Use ViZDoom's built-in scenarios
            config_path = Path(vzd.scenarios_path) / config_file

        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        self.game.load_config(str(config_path))

        # Set screen resolution
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        # Window visibility
        self.game.set_window_visible(visible)

        # Set rendering options
        self.game.set_render_hud(True)
        self.game.set_render_crosshair(True)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(True)
        self.game.set_render_particles(True)

        # Initialize game
        self.game.init()

        # Game state tracking
        self.episode_return = 0.0
        self.episode_length = 0

        # Previous state for computing rewards
        self.prev_health = None
        self.prev_armor = None
        self.prev_ammo = None
        self.prev_killcount = None
        self.prev_position = None

    def _get_observation(self) -> np.ndarray:
        """Get current frame as observation"""
        state = self.game.get_state()

        if state is None:
            # Return black screen if no state
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Get screen buffer
        screen = state.screen_buffer

        # Resize if needed
        if screen.shape[:2] != (self.height, self.width):
            screen = cv2.resize(screen, (self.width, self.height))

        return screen

    def _get_game_variables(self) -> Dict[str, float]:
        """Get game variables for reward computation"""
        state = self.game.get_state()

        if state is None:
            return {
                'health': 0,
                'armor': 0,
                'ammo': 0,
                'killcount': 0,
                'position_x': 0,
                'position_y': 0,
            }

        game_vars = state.game_variables

        # ViZDoom variables (depends on scenario config)
        # Common variables: health, armor, ammo, killcount
        variables = {
            'health': game_vars[0] if len(game_vars) > 0 else 0,
            'armor': game_vars[1] if len(game_vars) > 1 else 0,
            'ammo': game_vars[2] if len(game_vars) > 2 else 0,
            'killcount': game_vars[3] if len(game_vars) > 3 else 0,
            'position_x': game_vars[4] if len(game_vars) > 4 else 0,
            'position_y': game_vars[5] if len(game_vars) > 5 else 0,
        }

        return variables

    def _compute_reward(self, game_vars: Dict[str, float]) -> float:
        """
        Compute reward based on game variables
        Basic reward function (will be replaced by full Appendix A.5 version)
        """
        # Default: use game's built-in reward
        reward = self.game.get_last_reward()

        return reward

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # New episode
        self.game.new_episode()

        self.episode_return = 0.0
        self.episode_length = 0

        # Get initial observation
        obs = self._get_observation()

        # Initialize previous state
        game_vars = self._get_game_variables()
        self.prev_health = game_vars['health']
        self.prev_armor = game_vars['armor']
        self.prev_ammo = game_vars['ammo']
        self.prev_killcount = game_vars['killcount']
        self.prev_position = (game_vars['position_x'], game_vars['position_y'])

        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in environment"""

        # Convert action index to button presses
        action_vector = self.ACTIONS[action]

        # Apply action for frame_skip frames (paper uses 4)
        total_reward = 0.0

        for _ in range(self.frame_skip):
            reward = self.game.make_action(action_vector)
            total_reward += reward

            if self.game.is_episode_finished():
                break

        # Get observation
        obs = self._get_observation()

        # Check if episode is done
        terminated = self.game.is_episode_finished()
        truncated = False

        # Get game variables
        game_vars = self._get_game_variables()

        # Compute reward (using basic reward for now)
        reward = self._compute_reward(game_vars)

        # Update tracking
        self.episode_return += reward
        self.episode_length += 1

        # Update previous state
        self.prev_health = game_vars['health']
        self.prev_armor = game_vars['armor']
        self.prev_ammo = game_vars['ammo']
        self.prev_killcount = game_vars['killcount']
        self.prev_position = (game_vars['position_x'], game_vars['position_y'])

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info"""
        game_vars = self._get_game_variables()

        return {
            'episode_return': self.episode_return,
            'episode_length': self.episode_length,
            'health': game_vars['health'],
            'armor': game_vars['armor'],
            'ammo': game_vars['ammo'],
            'killcount': game_vars['killcount'],
        }

    def render(self):
        """Render environment"""
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            obs = self._get_observation()
            cv2.imshow('ViZDoom', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return obs
        return None

    def close(self):
        """Clean up resources"""
        self.game.close()


class ViZDoomEnvWithPaperReward(ViZDoomEnv):
    """
    ViZDoom Environment with Paper's Reward Function
    Implements reward function from Appendix A.5
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional tracking for paper's reward
        self.prev_items_count = 0
        self.prev_secrets_found = 0
        self.visited_positions = set()
        self.prev_fragcount = 0

    def _compute_reward(self, game_vars: Dict[str, float]) -> float:
        """
        Paper's reward function from Appendix A.5:

        1. Player hit: -100 points
        2. Player death: -5,000 points
        3. Enemy hit: 300 points
        4. Enemy kill: 1,000 points
        5. Item/weapon pick up: 100 points
        6. Secret found: 500 points
        7. New area: 20 * (1 + 0.5 * L1 distance) points
        8. Health delta: 10 * delta points
        9. Armor delta: 10 * delta points
        10. Ammo delta: 10 * max(0, delta) + min(0, delta) points
        """
        reward = 0.0

        # Get current variables
        health = game_vars['health']
        armor = game_vars['armor']
        ammo = game_vars['ammo']
        killcount = game_vars['killcount']
        pos_x = game_vars['position_x']
        pos_y = game_vars['position_y']

        # 1. Player hit (health decreased)
        if self.prev_health is not None and health < self.prev_health:
            health_lost = self.prev_health - health
            reward -= 100  # Penalty for getting hit

        # 2. Player death
        if health <= 0:
            reward -= 5000

        # 3 & 4. Enemy hits and kills
        if self.prev_killcount is not None:
            kills = killcount - self.prev_killcount
            if kills > 0:
                reward += 1000 * kills  # Enemy kill reward
                reward += 300 * kills   # Also count as hits

        # 5. Item/weapon pickup (heuristic: health or armor increase)
        if self.prev_health is not None and health > self.prev_health:
            reward += 100  # Item pickup bonus

        if self.prev_armor is not None and armor > self.prev_armor:
            reward += 100  # Armor pickup bonus

        # 7. New area exploration
        current_pos = (int(pos_x / 100), int(pos_y / 100))  # Grid cells
        if current_pos not in self.visited_positions:
            self.visited_positions.add(current_pos)

            # L1 distance from origin
            distance = abs(pos_x) + abs(pos_y)
            reward += 20 * (1 + 0.5 * distance / 1000.0)  # Scale distance

        # 8. Health delta
        if self.prev_health is not None:
            health_delta = health - self.prev_health
            reward += 10 * health_delta

        # 9. Armor delta
        if self.prev_armor is not None:
            armor_delta = armor - self.prev_armor
            reward += 10 * armor_delta

        # 10. Ammo delta
        if self.prev_ammo is not None:
            ammo_delta = ammo - self.prev_ammo
            reward += 10 * max(0, ammo_delta) + min(0, ammo_delta)

        return reward

    def reset(self, seed=None, options=None):
        """Reset with additional tracking"""
        obs, info = super().reset(seed, options)

        self.visited_positions.clear()
        self.prev_items_count = 0
        self.prev_secrets_found = 0
        self.prev_fragcount = 0

        # Add starting position
        game_vars = self._get_game_variables()
        current_pos = (int(game_vars['position_x'] / 100), int(game_vars['position_y'] / 100))
        self.visited_positions.add(current_pos)

        return obs, info


def create_vizdoom_env(
    scenario: str = "basic",
    width: int = 320,
    height: int = 256,
    frame_skip: int = 4,
    use_paper_reward: bool = False,
    visible: bool = False,
) -> gym.Env:
    """
    Factory function to create ViZDoom environment

    Args:
        scenario: Scenario name ('basic', 'deadly_corridor', etc.)
        width: Frame width
        height: Frame height
        frame_skip: Apply each action for N frames
        use_paper_reward: Use paper's reward function (Appendix A.5)
        visible: Show game window

    Returns:
        ViZDoom environment
    """
    config_file = f"{scenario}.cfg"

    if use_paper_reward:
        env = ViZDoomEnvWithPaperReward(
            config_file=config_file,
            width=width,
            height=height,
            frame_skip=frame_skip,
            visible=visible,
        )
    else:
        env = ViZDoomEnv(
            config_file=config_file,
            width=width,
            height=height,
            frame_skip=frame_skip,
            visible=visible,
        )

    return env


if __name__ == "__main__":
    # Test ViZDoom environment
    print("Testing ViZDoom Environment...")

    try:
        env = create_vizdoom_env(scenario="basic", visible=False)

        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Test reset
        obs, info = env.reset()
        print(f"\nReset observation shape: {obs.shape}")
        print(f"Info: {info}")

        # Test a few steps
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {i}: action={action}, reward={reward:.2f}, done={terminated}")

            if terminated:
                print("Episode ended")
                break

        env.close()
        print("\n✓ ViZDoom environment test passed!")

    except Exception as e:
        print(f"\n✗ ViZDoom test failed: {e}")
        print("\nMake sure ViZDoom is installed:")
        print("  pip install vizdoom")
        import traceback
        traceback.print_exc()
