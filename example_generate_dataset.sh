#!/bin/bash

# Example script to generate dataset from trained policy
# Usage: ./example_generate_dataset.sh

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Configuration
ENV_NAME="PointMaze_MediumDense-v3"
SNAPSHOT_PATH="/home/mprattico/url_benchmark_maze/exp_local/2025.09.25/092037_rnd/models/pixels/gym/rnd/1/snapshot.pt"  # Update this path
NUM_EPISODES=50
OUTPUT_DIR="./generated_datasets/rnd_medium"
SEED=42

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Generating dataset with the following configuration:"
echo "Environment: $ENV_NAME"
echo "Snapshot: $SNAPSHOT_PATH"
echo "Episodes: $NUM_EPISODES"
echo "Output: $OUTPUT_DIR"
echo "Seed: $SEED"
echo ""

# Run the dataset generation
python generate_dataset.py \
    --env_name "$ENV_NAME" \
    --snapshot_path "$SNAPSHOT_PATH" \
    --num_episodes "$NUM_EPISODES" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --frame_stack 3 \
    --action_repeat 2 \
    --resolution 84 \
    --device cuda \
    --random_init \
    --randomize_goal \
    --save_video

echo "Dataset generation completed!"
echo "Check $OUTPUT_DIR for the generated files."
