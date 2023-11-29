import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import torch.nn.functional as F
import pickle

class Agent:
    def __init__(self, board_size=10, frames=2, buffer_size=80000,
                 gamma=0.98, n_actions=3, use_target_net=True,
                 version=''):
        self._board_size = board_size
        self._frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._version = version
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize replay buffer (assuming a ReplayBuffer class exists)
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, self._frames, self._n_actions)
        
        # Initialize the model and target network if used (to be defined in subclasses)
        self._model = None
        self._target_net = None
        
        # Initialize optimizer (to be defined in subclasses)
        self._optimizer = None

    def save_buffer(self, file_path='', iteration=None):
        """Saves the replay buffer to a file."""
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open(f"{file_path}/buffer_{iteration:04d}.pkl", 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Loads the replay buffer from a file."""
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open(f"{file_path}/buffer_{iteration:04d}.pkl", 'rb') as f:
            self._buffer = pickle.load(f)

    def get_gamma(self):
        """Returns the gamma value."""
        return self._gamma

    def _point_to_row_col(self, point):
        """Converts a point into row and column coordinates."""
        return divmod(point, self._board_size)

    def _row_col_to_point(self, row, col):
        """Converts row and column coordinates into a point."""
        return row * self._board_size + col

    def reset_buffer(self, buffer_size=None):
        """Resets the replay buffer with a new size if provided."""
        if buffer_size is not None:
            self._buffer_size = buffer_size
        # Assuming the ReplayBuffer class can be re-initialized
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, self._frames, self._n_actions)

    def get_buffer_size(self):
        """Returns the current size of the replay buffer."""
        # Assuming the ReplayBuffer has a method to get its current size
        return self._buffer.get_current_size()

    def add_to_buffer(self, state, action, reward, next_state, done, next_legal_moves=None):
        """Adds an experience to the replay buffer."""
        if next_legal_moves is not None:
            # If next_legal_moves is provided, pass it to the buffer
            self._buffer.add_to_buffer(state, action, reward, next_state, done, next_legal_moves)
        else:
            # If next_legal_moves is not provided, use a default value (e.g., all zeros)
            legal_moves_default = np.zeros((1, self._n_actions))
            self._buffer.add_to_buffer(state, action, reward, next_state, done, legal_moves_default)


# Define the Q-Network for DeepQLearningAgent in PyTorch
class QNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(QNetwork, self).__init__()
        # Define a sequence of convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out(input_shape)

        # Define two separate streams for the dueling architecture
        # Stream for state value
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs a single value for the state value
        )

        # Stream for advantage
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)  # Outputs one value per action (the advantage of each action)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if x.size(1) != self.conv[0].in_channels:
            x = x.permute(0, 3, 1, 2)

        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)

        # Calculate the value and advantage streams
        value = self.fc_value(conv_out)
        advantage = self.fc_advantage(conv_out)

        # Combine the value and advantage to get the final Q values
        # Here, we subtract the mean of the advantage to stabilize training
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class DeepQLearningAgent(Agent):  # Inherits from Agent
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.98, n_actions=3, use_target_net=True,
                 version='', lr=0.0007):
        super().__init__(board_size, frames, buffer_size,  # Initialize the base class
                         gamma, n_actions, use_target_net,
                         version)

        self._model = self._get_model()  # DQN-specific model
        self._model.to(self._device)
        
        if self._use_target_net:
            self._target_net = self._get_model()
            self._target_net.to(self._device)
            self._target_net.load_state_dict(self._model.state_dict())
            self._target_net.eval()  # Set target network to evaluation mode

        learning_rate = 0.00001

        self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate, weight_decay=1e-5)


    def _get_model(self):
        # This method now uses the inherited properties from the Agent class
        return QNetwork((self._frames, self._board_size, self._board_size), self._n_actions)

    def _prepare_input(self, s):
        # Normalize the board state if necessary
        s_normalized = self._normalize_board(s)
        
        # Convert to a PyTorch tensor
        s_tensor = torch.tensor(s_normalized, dtype=torch.float32).to(self._device)
        
        # Permute the dimensions if needed to match [batch, channels, height, width]
        if s_tensor.ndim == 4 and s_tensor.shape[-1] in [self._frames, 2]:  # Assumes last dim is channels
            s_tensor = s_tensor.permute(0, 3, 1, 2)
        
        return s_tensor
    
    def _get_max_output(self):
        """Get the maximum output of Q values from the model."""
        s, _, _, _, _, _ = self._buffer.sample(self._buffer.get_current_size())
        s = self._prepare_input(s)
        max_value = self._model(s).max().item()
        return max_value

    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the loss."""
        s, a, r, next_s, done, _ = self._buffer.sample(batch_size)

        # Prepare the inputs
        s = self._normalize_board(s)  # Normalize if needed
        s = self._prepare_input(s)
        next_s = self._prepare_input(next_s)

        # Convert 'a' from NumPy array to PyTorch tensor if it's not already
        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.long).to(self._device)
        else:
            # Ensure it's a long type tensor and on the correct device
            a = a.to(dtype=torch.long, device=self._device)

        # Convert 'r' and 'done' to PyTorch tensors
        r = torch.tensor(r, dtype=torch.float32).to(self._device).squeeze(1)
        done = torch.tensor(done, dtype=torch.float32).to(self._device).squeeze(1)

        # Get the Q values for current states
        q_values = self._model(s)

        # If 'a' is one-hot encoded or not in the correct shape, convert it to indices
        if a.ndim > 1 and a.shape[1] > 1:
            a = torch.argmax(a, dim=1)

        # Unsqueeze 'a' to match the Q-values dimension for gather
        a = a.unsqueeze(-1)

        # Gather the Q values corresponding to the actions taken
        q_values = q_values.gather(1, a).squeeze(-1)

        # Get the next Q values using the target net if specified, otherwise use the current model
        next_q_values = self._target_net(next_s).detach().max(1)[0] if self._use_target_net else self._model(next_s).detach().max(1)[0]

        # Compute the expected Q values
        expected_q_values = r + self._gamma * next_q_values * (1 - done)

        # Ensure expected_q_values is a 1D tensor [batch_size]
        expected_q_values = expected_q_values.squeeze(-1) 

        # Compute the loss using Huber Loss
        loss = nn.SmoothL1Loss()(q_values, expected_q_values)
                
        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _update_target(self):
        """Update the target network by copying the model's weights."""
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())


    def save_model(self, file_path="models/v17.1/model", iteration=None):
        """Save the model weights to the specified path."""
        if iteration is not None:
            # If an iteration number is provided, include it in the filename
            full_path = f"{file_path}_iteration_{iteration}.pth"
        else:
            # Otherwise, just use the file_path with .pth extension
            full_path = f"{file_path}.pth"
 
        torch.save(self._model.state_dict(), full_path)


    def load_model(self, file_path, iteration=None):
        """Load the model weights from the specified path."""
        if iteration is not None:
            # If an iteration number is provided, include it in the filename
            full_path = f"{file_path}_iteration_{iteration}.pth"
        else:
            # Otherwise, just use the file_path with .pth extension
            full_path = f"{file_path}.pth"
 
        self._model.load_state_dict(torch.load(full_path))
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def _normalize_board(self, board):
        # Assuming the board is a NumPy array that needs to be normalized before conversion to a tensor.
        # Adjust the normalization to match the TensorFlow version if needed.
        return (board / 4.0).astype(np.float32)
    
    def move(self, board, legal_moves, values=None):
        # Ensure 'board' is a numpy array
        assert isinstance(board, np.ndarray), "Board must be a numpy array"
        # Add batch dimension if it's missing
        if len(board.shape) == 3:
            board = np.expand_dims(board, axis=0)
        assert len(board.shape) == 4, "Board must be 4D with shape [batch, height, width, channels]"

        # Prepare the state tensor by converting the numpy array to a PyTorch tensor
        # and moving it to the correct device
        state_tensor = torch.tensor(board, dtype=torch.float32).to(self._device)

        # Permute the dimensions to match [batch, channels, height, width]
        state_tensor = state_tensor.permute(0, 3, 1, 2)

        # Pass the state tensor to the model
        with torch.no_grad():
            q_values = self._model(state_tensor)

        # Convert q_values to a numpy array if it's a tensor
        q_values_np = q_values.cpu().numpy()

        # Ensure 'legal_moves' is a numpy array and has the correct shape
        assert isinstance(legal_moves, np.ndarray), "Legal moves must be a numpy array"
        assert legal_moves.shape == (board.shape[0], self._n_actions), \
            "Legal moves should have shape [batch, n_actions]"

        # Mask the illegal actions by setting their Q values to -inf
        masked_q_values = np.where(legal_moves, q_values_np, -np.inf)

        # Select the action with the highest Q value for each batch
        return np.argmax(masked_q_values, axis=1)

    
    def update_target_net(self):
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())

    def get_action_proba(self, state):
        # This function computes the action probabilities using softmax.
        state_tensor = self._prepare_input(np.array([state]))
        with torch.no_grad():
            q_values = self._model(state_tensor)
            action_probabilities = F.softmax(q_values, dim=1)
        return action_probabilities.cpu().numpy()
    
    def _get_model_outputs(self, state):
        # Prepare the input for the model
        state_tensor = self._prepare_input(state)
        # Get the model's prediction
        with torch.no_grad():
            model_outputs = self._model(state_tensor).cpu().numpy()
        return model_outputs

