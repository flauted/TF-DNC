# TF-DNC [Work in Progress]
A Tensorflow implementation of DeepMind's Differentiable Neural Computer.

HTML documentation guide:
* Navigate to `docs/_build/html` and open in your preferred browser.
* The DNC documentation covers object usage. The memory documentation covers most of the bulk of formula implementation.
* The documentation and the illustrated .pdf paper aim to explain the DNC architecture with multiple write heads. If you wish to follow this DNC implementation to help write your own code with only one write head (as in the paper), let *H = 1*.

The goals of this repo are as follows: Provide an example of a DNC with a **Convolutional** Neural Network as controller, discuss the math behind the DNC model, implement the DNC as a TensorFlow RNN object while being accessible for those who are not familiar with the RNN API, offer OOP code without relying on attributes to pass arguments, provide a **TensorBoard** graph, incorporate **softmax addressing** and the original implementation, include **multiple write heads** with discussion, and provide a **stateful** option.

## Implementation and Tasks
We inherit from the TensorFlow RNN class as suggested in [5]. This allows flexibility and built-in parallelization. The model code is divided between the DNC class and the `Memory` class, which inherits from `DNC`. State variables are passed in an `AccessState` named-tuple defined in `memory.py`. This mechanism is similar to the one used in the original code [3]. From [1] and [2], we borrow the idea of updating the usage with a weighted softmax. The user chooses original implementation or softmax as a parameter to `DNC.__init__()`. Additionally, we create a stateful implementation option within the task training. We use the multiple write heads that are implemented and discussed originally in the DeepMind source code [3].

As for tasks, we notice that few public models use convolutional layers for image tasks. We have implemented a convolutional DNC. It takes a sequence of images and is tasked with replicating one hot image classification. It is similar to the standard copy task in [3], [6], and [7], but with the added challenge of recognizing MNIST digits. (As of now, this assumes a local download of MNIST training images and labels.)

## Usage
Download or clone the repo. Run

    $ python CopyTask.py

to train on the standard copy task. Use

    $ python CopyTask.py -h

to view command line arguments and default values.

To run the MNIST convolutional recognition and copy task, run

    $ python MNISTCopyTask.py

or invoke the `-h` flag to view parameters.

Be aware, the `*Task.py` scripts default to writing a TensorBoard logdir at `tb/dnc`. The location may be changed at the command line. The CopyTask has a single accumulator. Since data is randomly generated, there is no need for testing. The MNISTCopyTask has two - one for training and one for testing. The names of the accumulators may be changed at the command line for either script. The files are not deleted upon running either script, so delete the logdir if you do not want loss curves from different script invocations on the same axes!

## Future Features
* Looking for ways to improve graph visualization. Suggestions are welcome!
* Better task documentation.
* Variable sequence length.

## Sources

[1]: Albert, J., "Issue 21, deepmind/dnc"

[2]: Ben-Ari, I., Bekkar, A., "Differentiable Memory Allocation Mechanism for Neural Computing"

[3]: "deepmind/dnc", Github.

[4]: Graves, A. et al., "Hybrid computing using a neural network with dynamic external memory"

[5]: Hsin, C., "Implementation and Optimization of Differentiable Neural Computers"

[6]: Raval, S., "llSourcell/differentiable_neural_computer_LIVE", Github.

[7]: Samir, M., "Mostafa-Samir/DNC-tensorflow", Github.
