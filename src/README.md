### Observations

After running `tf_vit.ipynb` or `keras_vit.ipynb`, which may consume almost 100% of GPU memory and doesn't seem to release it, make sure to: (1) `Shut Down All Kernels...` and (2) `Restart Kernel...` to clear the GPU memory. Otherwise, you might encounter `out of memory (OOM)` related errors when subsequently running other code. That's my temporary workaround that I came up with to avoid memory issues when running the Tensorflow and Keras ViT implemetations. BTW, the test runs were carried out on A100 with 80GB of memory.

Interestingly, the TensorFlow/Keras implementations ran faster than the PyTorch implementations, even with the same datasets being fed. Regarding this performance difference, it was observed that the PyTorch implementations consumed only about half of the GPU memory, while the TensorFlow/Keras implementations utilized the entire GPU memory.
