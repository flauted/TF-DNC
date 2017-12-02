# TF-DNC [Work in Progress]
A Tensorflow implementation of DeepMind's Differentiable Neural Computer.

HTML documentation guide:
* Navigate to `docs/_build/html` and open in your preferred browser.
* The DNC documentation covers usage. The memory documentation covers most of the implementation.
* The documentation aims to serve as a companion/explainer for the Graves paper. Formulae are replicated in LaTeX, then implementation is discussed.

The goals of this repo are as follows: Provide an example of a DNC with a **Convolutional** Neural Network as controller, discuss the math behind the DNC model, implement the DNC as a TensorFlow RNN object while being accessible for those who are not familiar with the RNN API, offer OOP code without relying on attributes to pass arguments, provide a **TensorBoard** graph, incorporate **softmax addressing** and the original implementation, and provide a **stateful** option.

## Implementation and Tasks
We inherit from the TensorFlow RNN class as suggested in [5]. This allows flexibility and built-in parallelization. The model code is divided between the DNC class and the `Memory` class, which inherits from `DNC`. State variables are passed in an `AccessState` named-tuple defined in `memory.py`. This mechanism is similar to the one used in the original code [3]. From [1] and [2], we borrow the idea of updating the usage with a weighted softmax. The user chooses original implementation or softmax as a parameter to `DNC.__init__()`. Additionally, we create a stateful implementation option within the task training.

As for tasks, we notice that few public models use convolutional layers for image tasks. We have implemented a convolutional DNC. It takes a sequence of images and is tasked with replicating one hot image classification. It is similar to the standard copy task in [3], [6], and [7], but with the added challenge of recognizing MNIST digits. (As of now, this assumes a local download of MNIST training images and labels.)

## Usage
Download or clone the repo. Run

    $ python CopyTask.py

to train on the standard copy task. (Command line flags are a planned feature; currently parameters must be set in the `*Task.py` source.)

Be aware, the `*Task.py` scripts default to writing a TensorBoard logdir at `tb/dnc`.

## Future Features
* Looking for ways to improve graph visualization. Suggestions are welcome!
* Command line args (soon).
* Better task documentation.
* Variable sequence length.
* Perhaps multiple write heads.

## Sources

[1]: Albert, J., "Issue 21, deepmind/dnc"

[2]: Ben-Ari, I., Bekkar, A., "Differentiable Memory Allocation Mechanism for Neural Computing"

[3]: "deepmind/dnc", Github.

[4]: Graves, A. et al., "Hybrid computing using a neural network with dynamic external memory"

[5]: Hsin, C., "Implementation and Optimization of Differentiable Neural Computers"

[6]: Raval, S., "llSourcell/differentiable_neural_computer_LIVE", Github.

[7]: Samir, M., "Mostafa-Samir/DNC-tensorflow", Github.
