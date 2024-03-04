from nn import *


def train(n, episodes=20, learning_rate=-0.1):
    """
    n: the neural network
    episodes: the times of the training
    learning_rate: the step size of each gradient descent step
    """
    for k in range(episodes):

        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum(
            (yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)
        )  # The loss function is simply the MSE loss.

        # backward pass
        # n.zero_grad()
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -learning_rate * p.grad

        print(k, loss.data)


if __name__ == "__main__":
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(3, [4, 4, 1])
    train(n)
