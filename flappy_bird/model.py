import torch
# import random
# import os


class NeuralNet(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.linear3(x)
        return x

    def save(self, path_to_save: str):
        torch.save(self.state_dict(), path_to_save)
        pass

    def load(self, path_to_file: str):
        self.load_state_dict(torch.load(path_to_file))


class QLearning:

    def __init__(self, model: NeuralNet, target_model: NeuralNet, lr: float, gamma: float) -> None:
        self.model = model
        self.target_model = target_model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), self.lr)
        self.criterion = torch.nn.MSELoss()

    def train(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        new_state: torch.Tensor,
        done: torch.Tensor,
    ):
        '''
        All the state, action, reward, new_state will be of shape (n, x)
        Where n is number of data points and x is used for representation
        '''

        prediction: torch.Tensor = self.model(state)  # (n, action_size)

        target: torch.Tensor = prediction.clone()

        with torch.no_grad():
            best_next_actions = self.model(new_state).argmax(dim=1).unsqueeze(1)
            Q_new_next_state = self.target_model(new_state).gather(1, best_next_actions)
        done_bool = done.bool() if done.dtype == torch.float else done.bool()
        Q_target_values = reward.clone()
        Q_new_next_state = Q_new_next_state.reshape((-1, 1))
        Q_target_values += self.gamma * Q_new_next_state*(~done_bool)
        action_indices = action.long().squeeze()
        target[..., action_indices] = Q_target_values

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        self.optimizer.step()  # type: ignore
