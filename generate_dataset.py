#!/usr/bin/env python3

import argparse
import os
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from dm_env import specs

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set environment variables
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'osmesa'

# Import local modules
import gym_env
import utils
from replay_buffer import save_episode
from video import VideoRecorder


def load_agent_from_snapshot(snapshot_path):
    """Load agent from snapshot file."""
    if snapshot_path is None or snapshot_path.lower() == "none":
        return None
        
    snapshot = Path(snapshot_path)
    if not snapshot.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
    
    print(f"Loading agent from snapshot: {snapshot_path}")
    with snapshot.open('rb') as f:
        # Set weights_only=False to allow loading custom agent classes
        payload = torch.load(f, map_location='cpu', weights_only=False)
    
    if 'agent' not in payload:
        raise KeyError("Agent not found in snapshot file")
    
    return payload['agent']


class RandomAgent:
    """Simple random agent for data collection when no snapshot is provided."""
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.device = 'cpu'
    
    def get_meta_specs(self):
        return tuple()
    
    def init_meta(self):
        from collections import OrderedDict
        return OrderedDict()
    
    def update_meta(self, meta, global_step, time_step):
        return meta
    
    def act(self, obs, meta, step, eval_mode=True):
        return self.action_space.sample()
    
    def eval(self):
        pass
    
    def to(self, device):
        self.device = device
        return self


def create_replay_buffer_storage(data_specs, meta_specs, output_dir):
    """Create a simple storage class to save episodes."""
    class SimpleStorage:
        def __init__(self, data_specs, meta_specs, replay_dir):
            self._data_specs = data_specs
            self._meta_specs = meta_specs
            self._replay_dir = Path(replay_dir)
            self._replay_dir.mkdir(exist_ok=True, parents=True)
            self._episode_count = 0
            print(f"Created storage directory: {self._replay_dir}")
        
        def save_episode_data(self, episode_data):
            """Save a complete episode to disk."""
            # Calculate episode length (excluding dummy first transition)
            eps_len = len(episode_data['observation']) - 1
            
            # Create episode dict with numpy arrays
            episode = {}
            for spec in self._data_specs:
                if spec.name in episode_data:
                    episode[spec.name] = np.array(episode_data[spec.name], dtype=spec.dtype)
            
            for spec in self._meta_specs:
                if spec.name in episode_data:
                    episode[spec.name] = np.array(episode_data[spec.name], dtype=spec.dtype)
            
            # Create filename
            eps_idx = self._episode_count
            ts = "generated"  # Simple timestamp replacement
            eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
            
            # Save episode
            file_path = self._replay_dir / eps_fn
            save_episode(episode, file_path)
            
            self._episode_count += 1
            print(f"Saved episode {eps_idx} with {eps_len} transitions to {eps_fn}")
            
            return eps_len
    
    return SimpleStorage(data_specs, meta_specs, output_dir)


def collect_episodes(env, agent, num_episodes, storage, video_recorder=None, device='cuda'):
    """Collect episodes using the trained agent."""
    print(f"Starting episode collection with device: {device}")
    
    # Move agent to device if possible
    try:
        agent = agent.to(device)
    except AttributeError:
        # Some agents might not have .to() method, try to move individual components
        print("Agent doesn't have .to() method, trying to move individual components...")
        if hasattr(agent, 'device'):
            agent.device = torch.device(device)
        
        # Try to move common agent components to device
        for attr_name in ['encoder', 'actor', 'critic', 'critic_target']:
            if hasattr(agent, attr_name):
                attr = getattr(agent, attr_name)
                if hasattr(attr, 'to'):
                    setattr(agent, attr_name, attr.to(device))
                    print(f"Moved {attr_name} to {device}")
    
    # Set agent to evaluation mode if possible
    try:
        agent.eval()
    except AttributeError:
        print("Agent doesn't have .eval() method, continuing without it...")
    
    total_transitions = 0
    episode_rewards = []
    
    for episode_idx in range(num_episodes):
        print(f"\nCollecting episode {episode_idx + 1}/{num_episodes}")
        
        # Reset environment and agent meta
        time_step = env.reset()
        meta = agent.init_meta()
        
        # Initialize episode storage
        episode_data = defaultdict(list)
        
        # Add first dummy transition (required by replay buffer format)
        for spec_name in ['observation', 'action', 'reward', 'discount', 'proprio_observation']:
            if spec_name == 'observation':
                episode_data[spec_name].append(time_step.observation)
            elif spec_name == 'action':
                episode_data[spec_name].append(np.zeros_like(env.action_space.sample()))
            elif spec_name == 'reward':
                episode_data[spec_name].append(np.array([0.0], dtype=np.float32))
            elif spec_name == 'discount':
                episode_data[spec_name].append(np.array([1.0], dtype=np.float32))
            elif spec_name == 'proprio_observation':
                episode_data[spec_name].append(time_step.proprio_observation)
        
        # Add meta data for first transition
        for key, value in meta.items():
            episode_data[key].append(value)
        
        # Initialize video recorder if provided
        if video_recorder:
            video_recorder.init(env, enabled=True)
        
        episode_reward = 0
        step_count = 0
        
        # Collect episode
        while not time_step.last():
            # Sample action from agent
            with torch.no_grad():
                if hasattr(agent, 'act'):
                    # Direct agent action call
                    action = agent.act(time_step.observation, meta, 0, eval_mode=True)
                else:
                    # Convert observation to tensor and call agent
                    obs_tensor = torch.from_numpy(time_step.observation).float().unsqueeze(0).to(device)
                    action = agent(obs_tensor, meta, 0, eval_mode=True)
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim > 1:
                    action = action.squeeze(0)
            
            # Take environment step
            time_step = env.step(action)
            
            # Update meta if agent has update_meta method
            if hasattr(agent, 'update_meta'):
                meta = agent.update_meta(meta, 0, time_step)
            
            # Store transition
            episode_data['observation'].append(time_step.observation)
            episode_data['action'].append(action)
            episode_data['reward'].append(np.array([time_step.reward], dtype=np.float32))
            episode_data['discount'].append(np.array([time_step.discount], dtype=np.float32))
            episode_data['proprio_observation'].append(time_step.proprio_observation)
            
            # Store meta data
            for key, value in meta.items():
                episode_data[key].append(value)
            
            # Record video frame
            if video_recorder:
                video_recorder.record(env)
            
            episode_reward += time_step.reward
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}, reward: {episode_reward:.2f}")
        
        # Save episode
        episode_length = storage.save_episode_data(episode_data)
        total_transitions += episode_length
        episode_rewards.append(episode_reward)
        
        # Save video
        if video_recorder:
            video_recorder.save(f'episode_{episode_idx}.mp4')
        
        print(f"Episode {episode_idx + 1} completed: {episode_length} transitions, reward: {episode_reward:.2f}")
    
    print(f"\nDataset collection completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total transitions: {total_transitions}")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {total_transitions / num_episodes:.1f}")
    
    return total_transitions, episode_rewards


def main():
    parser = argparse.ArgumentParser(description='Generate dataset from trained policy')
    parser.add_argument('--env_name', type=str, required=True,
                       help='Environment name (e.g., PointMaze_Medium-v3)')
    parser.add_argument('--snapshot_path', type=str, default=None,
                       help='Path to the agent snapshot file (if None, uses random policy)')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--output_dir', type=str, default='./generated_dataset',
                       help='Directory to save the dataset')
    parser.add_argument('--frame_stack', type=int, default=1,
                       help='Number of frames to stack')
    parser.add_argument('--action_repeat', type=int, default=1,
                       help='Action repeat factor')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    parser.add_argument('--resolution', type=int, default=224,
                       help='Image resolution')
    parser.add_argument('--random_init', action='store_true', default=True,
                       help='Use random initial positions')
    parser.add_argument('--randomize_goal', action='store_true', default=True,
                       help='Use random goals')
    parser.add_argument('--save_video', action='store_true',
                       help='Save videos of episodes')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    utils.set_seed_everywhere(args.seed)
    
    # Load agent from snapshot or create random agent
    agent = load_agent_from_snapshot(args.snapshot_path)
    
    if agent is None or agent == "None" or agent == "none":
        # Print warning in yellow and use random agent
        print("\033[93m⚠️  No snapshot provided, using random policy for data collection\033[0m")
        # Create environment first to get action space
        temp_env = gym_env.make(
            name=args.env_name,
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            seed=args.seed,
            resolution=args.resolution,
            random_init=args.random_init,
            randomize_goal=args.randomize_goal
        )
        agent = RandomAgent(temp_env.action_space)
        temp_env = None  # Clean up
    
    # Create environment
    env = gym_env.make(
        name=args.env_name,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        resolution=args.resolution,
        random_init=args.random_init,
        randomize_goal=args.randomize_goal
    )
    
    # Get specs
    obs_spec = gym_env.observation_spec(env)
    action_spec = gym_env.action_spec(env)
    
    # Get sample to determine proprio observation shape
    sample_time_step = env.reset()
    proprio_shape = sample_time_step.proprio_observation.shape
    
    # Define data specs (same as in pretrain_gym.py)
    data_specs = (
        obs_spec,
        action_spec,
        specs.Array((1,), np.float32, 'reward'),
        specs.Array((1,), np.float32, 'discount'),
        specs.Array(proprio_shape, np.float32, 'proprio_observation')
    )
    
    # Get meta specs from agent
    meta_specs = agent.get_meta_specs()
    
    # Create storage
    storage = create_replay_buffer_storage(data_specs, meta_specs, args.output_dir)
    
    # Create video recorder if requested
    video_recorder = None
    if args.save_video:
        video_dir = Path(args.output_dir) / 'videos'
        video_dir.mkdir(exist_ok=True, parents=True)
        video_recorder = VideoRecorder(
            video_dir,
            camera_id=0 if 'quadruped' not in args.env_name else 2,
            use_wandb=False
        )
    
    # Collect episodes
    total_transitions, episode_rewards = collect_episodes(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        storage=storage,
        video_recorder=video_recorder,
        device=device
    )
    
    # Save summary statistics
    output_dir = Path(args.output_dir)
    stats_file = output_dir / 'dataset_stats.txt'
    with open(stats_file, 'w') as f:
        f.write(f"Dataset Statistics\n")
        f.write(f"==================\n")
        f.write(f"Environment: {args.env_name}\n")
        f.write(f"Snapshot: {args.snapshot_path}\n")
        f.write(f"Episodes: {args.num_episodes}\n")
        f.write(f"Total transitions: {total_transitions}\n")
        f.write(f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}\n")
        f.write(f"Average episode length: {total_transitions / args.num_episodes:.1f}\n")
        f.write(f"Seed: {args.seed}\n")
    
    print(f"Dataset statistics saved to: {stats_file}")


if __name__ == '__main__':
    main()
