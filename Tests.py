#!/usr/bin/env python
import MNISTCopyTask
import CopyTask


if __name__ == "__main__":
    for softmax_alloc in [True, False]:
        for batch_size in [1, 5]:
            for num_read_heads in [1, 5]:
                try:
                    MNISTCopyTask.run_training(iterations=10,
                                               num_read_heads=num_read_heads,
                                               batch_size=batch_size,
                                               softmax_alloc=softmax_alloc)
                except:
                    print("MNIST Failed Test")
                    print("Softmax alloc?:", softmax_alloc)
                    print("Batch size:", batch_size)
                    print("Read heads:", num_read_heads)
                else:
                    print("MNIST passed.")

                try:
                    CopyTask.run_training(iterations=10,
                                          num_read_heads=num_read_heads,
                                          batch_size=batch_size,
                                          softmax_alloc=softmax_alloc)
                except:
                    print("CopyTask Failed Test")
                    print("Softmax alloc?:", softmax_alloc)
                    print("Batch size:", batch_size)
                    print("Read heads:", num_read_heads)
                else:
                    print("CopyTask passed.")
