import torch
import imageio
import os
import glob
import sys
import numpy as np

# Add project directories to the path to allow imports
sys.path.append("env")
sys.path.append("train")

from uav_env import UAVEnv
from dqnn_agent import DQNNAgent

def create_visualization(model_path, env, output_filename="trajectory.gif"):
    """
    Creates a GIF visualization of a trained agent's trajectory.
    """
    print(f"--- Creating visualization for model: {model_path} ---")

    # 1. Load the trained agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNNAgent(state_dim, action_dim, dueling=True, double=True)
    agent.q_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    agent.q_net.eval()

    # 2. Prepare a directory to save the image frames
    frame_dir = "frames"
    if os.path.exists(frame_dir):
        # Clean up old frames from a previous run
        files = glob.glob(os.path.join(frame_dir, "*.png"))
        for f in files:
            os.remove(f)
    else:
        os.makedirs(frame_dir)

    # 3. Run one episode and save a frame at each step
    state = env.reset()
    done = False
    step_count = 0
    while not done:
        action = agent.select_action(state, train_mode=False)
        state, _, done, _ = env.step(action)
        
        # Render the current state and save it as a PNG file
        env.render_2d(save=True, frame_id=step_count)
        step_count += 1
        if step_count > 200: # Safety break to prevent infinite loops
            print("Episode exceeded 200 steps. Stopping.")
            break
            
    print(f"Episode finished in {step_count} steps.")

    # 4. Create a GIF from the saved frames
    print("Stitching frames into a GIF...")
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    images = [imageio.imread(file) for file in frame_files]
    imageio.mimsave(output_filename, images, fps=5) # fps controls the speed of the GIF
    
    print(f"✅ GIF saved as {output_filename}")

    # 5. Clean up the temporary frame files
    for file in frame_files:
        os.remove(file)
    os.rmdir(frame_dir)
    print("Cleaned up temporary frame files.")


if __name__ == "__main__":
    # Load the UNSEEN environment you want to visualize
    # This uses the environment with 13 obstacles
    env = UAVEnv(data_dir="data_unseen") 
    
    # Specify the path to your best model
    # Use either your static model or the randomized one
    model_to_visualize = "best_model.pth" 
    
    # Run the visualization function
    create_visualization(model_to_visualize, env)
