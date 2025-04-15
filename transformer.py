import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28

def transformer(X: ad.Node, nodes: List[ad.Node], 
                      model_dim: int, seq_len: int, input_dim: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, input_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """
    class FeedForwardLayer:
        def __init__(self, W_1: ad.Node, W_2: ad.Node, b_1: ad.Node, b_2: ad.Node, batch_size: int, model_dim: int, num_classes: int, seq_len: int):
            self.W_1 = ad.broadcast(W_1, [model_dim, model_dim], [batch_size, model_dim, model_dim])
            self.W_2 = ad.broadcast(W_2, [model_dim, num_classes], [batch_size, model_dim, num_classes])
            self.b_1 = ad.broadcast(b_1, [model_dim], [batch_size, seq_len, model_dim])
            self.b_2 = ad.broadcast(b_2, [num_classes], [batch_size, seq_len, num_classes])
        
        def forware(self, X: ad.Node) -> ad.Node:
            X = ad.matmul(X, self.W_1) + self.b_1
            X = ad.relu(X)
            X = ad.matmul(X, self.W_2) + self.b_2
            return X

    class SingleHeadAttentionLayer:
        def __init__(self, W_Q: ad.Node, W_K: ad.Node, W_V: ad.Node, W_O: ad.Node, batch_size: int, input_dim: int, model_dim: int):
            self.query = ad.broadcast(W_Q, [input_dim, model_dim], [batch_size, input_dim, model_dim])
            self.key = ad.broadcast(W_K, [input_dim, model_dim], [batch_size, input_dim, model_dim])
            self.value = ad.broadcast(W_V, [input_dim, model_dim], [batch_size, input_dim, model_dim])
            self.output = ad.broadcast(W_O, [model_dim, model_dim], [batch_size, model_dim, model_dim])
            self.dk = model_dim ** 0.5

        def forward(self, X: ad.Node) -> ad.Node:
            q = ad.matmul(X, self.query)
            k = ad.matmul(X, self.key)
            v = ad.matmul(X, self.value)
            scores = ad.matmul(q, ad.transpose(k, -2, -1)) / self.dk
            attention_weights = ad.softmax(scores, dim=-1)
            output = ad.matmul(attention_weights, v)
            output = ad.matmul(output, self.output)
            return output
        
    singlehead = SingleHeadAttentionLayer(nodes[0], nodes[1], nodes[2], nodes[3], batch_size, input_dim, model_dim)
    feedforward = FeedForwardLayer(nodes[4], nodes[5], nodes[6], nodes[7], batch_size, model_dim, num_classes, seq_len)
    output = singlehead.forward(X)
    output = ad.layernorm(output, [model_dim], eps)
    output = feedforward.forware(output)
    output = ad.sum_op(output, dim=-2, keepdim=False)
    output = ad.layernorm(output, [num_classes], eps)
    return output


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    predictions = ad.softmax(Z, dim=-1)
    loss = ad.mul_by_const(ad.sum_op(y_one_hot * ad.log(predictions), dim=[0, 1], keepdim=False), -1) / batch_size
    return loss

def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes
        _, loss, grads = f_run_model(X_batch, y_batch, model_weights)
        for i in range(len(model_weights)):
            model_weights[i] -= lr * grads[i]
        total_loss += loss * batch_size

    # Compute the average loss    
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5 

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    x = ad.Variable(name="x")
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")
    b_2 = ad.Variable(name="b_2")
    parameters = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    y_predict: ad.Node = transformer(x, [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2], model_dim, seq_length, input_dim, eps, batch_size, num_classes) # The output of the forward pass
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    grads: List[ad.Node] = ad.gradients(loss, parameters)
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(X_batch, y_batch, model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        result = evaluator.run(
            input_values={
                x : X_batch,
                y_groundtruth : y_batch,
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],
            }
        )
        return result[0], result[1], result[2:]

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            
            logits = test_evaluator.run({
                x: X_batch,
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],                
            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [torch.tensor(W_Q_val), torch.tensor(W_K_val), torch.tensor(W_V_val), torch.tensor(W_O_val), torch.tensor(W_1_val), torch.tensor(W_2_val), torch.tensor(b_1_val), torch.tensor(b_2_val)]
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
