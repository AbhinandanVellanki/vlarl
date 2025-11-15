"""
Usage:
    pytest tests/test_torch.py -s
"""

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SimpleMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def test_torch_mlp_training():
    print("\nRunning simple MLP training test...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    input_size = 768
    hidden_size = 512
    output_size = 768
    num_layers = 10
    batch_size = 1024
    num_epochs = 1000000
    learning_rate = 0.001

    # Model, Loss, and Optimizer
    model = SimpleMLP(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy data
    dummy_input = torch.randn(batch_size, input_size).to(device)
    dummy_target = torch.randn(batch_size, output_size).to(device)

    for epoch in range(num_epochs):
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Simple MLP training test completed successfully.")