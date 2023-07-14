import torch as tc
from datasets import CIFAR10HePreprocessing
from model import Cifar10ResNet
from runners import Evaluator, Trainer
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10 with preprocessing as described in Section 4.2 of He et al., 2015.
training_data = CIFAR10HePreprocessing(root="data", train=True)
test_data = CIFAR10HePreprocessing(root="data", train=False)

batch_size = 128

# Create data loaders.
train_dataloader = tc.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = tc.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if tc.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# The ResNet-20 model from He et al., 2015 for CIFAR-10.
model = Cifar10ResNet(
    img_height=32, img_width=32, img_channels=3,
    initial_num_filters=16, num_repeats=3, num_stages=3, num_classes=10).to(device)
print(model)

try:
    model.load_state_dict(tc.load("model.pth"))
    print('successfully reloaded checkpoint. continuing training...')
except Exception:
    print('no checkpoint found. training from scratch...')

loss_fn = tc.nn.CrossEntropyLoss()

evaluator = Evaluator()
trainer = Trainer(max_iters=16000, evaluator=evaluator, verbose=True)

trainer.run(model, train_dataloader, test_dataloader, device, loss_fn)
print("Done!")

tc.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


model.eval() # turn batchnorm and dropout off.
for X, y in test_dataloader:
    input_example = X[0]
    input_label = y[0]

    x_features = model.visualize(
        tc.unsqueeze(input_example, dim=0))

    y_pred = tc.nn.Softmax()(model(tc.unsqueeze(input_example, dim=0)))

    print('ground truth label: {}'.format(input_label))
    print('predicted label distribution: {}'.format(y_pred))
    print(x_features)

    plt.imshow(np.transpose(input_example, [1,2,0]))
    plt.show()

    break
