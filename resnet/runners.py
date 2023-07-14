import torch as tc
import collections

class Evaluator:
    def __init__(self):
        self.foo = 0

    def run(self, model, test_dataloader, device, loss_fn):
        num_test_examples = len(test_dataloader.dataset)
        model.eval() # this turns batchnorm, dropout, etc. to eval mode.
        test_loss, correct = 0, 0
        with tc.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += len(X) * loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(tc.float).sum().item()
        test_loss /= num_test_examples
        correct /= num_test_examples
        return {
            "accuracy": correct,
            "loss": test_loss
        }


class Trainer:
    def __init__(self, max_iters, evaluator, verbose=True):
        self.max_iters = max_iters
        self.evaluator = evaluator
        self.verbose = verbose
        self.global_step = 0 # would be nice if we could checkpoint this like in tensorflow; look into later

    def run(self, model, train_dataloader, test_dataloader, device, loss_fn):

        epoch = 1
        optimizer = tc.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = tc.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000,48000], gamma=0.10)

        while self.global_step < self.max_iters:
            model.train() # turns batchnorm, dropout, etc. to train mode.

            if self.verbose:
                print(f"Epoch {epoch}\n-------------------------------")

            for (X, y) in train_dataloader:
                X, y = X.to(device), y.to(device)

                # Forward
                logits = model(X)
                loss = loss_fn(logits, y)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # update global step
                self.global_step += 1

                if self.global_step % 100 == 0 and self.verbose:
                    loss = loss.item()
                    print(f"loss: {loss:>7f}  [{self.global_step:>5d}/{self.max_iters:>5d}]")

            # after every epoch, print stats for test set. bad practice, should be validation set. fix later.
            eval_dict = self.evaluator.run(model, test_dataloader, device, loss_fn)
            accuracy = eval_dict['accuracy'] * 100
            test_loss = eval_dict['loss']
            if self.verbose:
                print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")

            epoch += 1
