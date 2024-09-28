


### Some Observations on TensorFlow, Keras, and PyTorch ViT Implementations
After running tf_vit.ipynb or keras_vit.ipynb, which may consume almost 100% of GPU memory and doesn't seem to release it, make sure to (on the pulldown menu of Jupyter notebook):

- Shut Down All Kernels...
- Restart Kernel...

These steps are needed to clear the GPU memory after running the TensorFlow/Keras implementations. Otherwise, you might encounter out of memory (OOM) related errors when subsequently running other code. This is my temporary workaround to avoid memory issues when running the TensorFlow and Keras ViT implementations. The test runs were carried out on an A100 GPU with 80GB of memory.

Interestingly, the TensorFlow/Keras implementations ran faster than the PyTorch implementations, even with the same datasets being fed. Regarding this performance difference, it was observed that the PyTorch implementations consumed only about half of the GPU memory, while the TensorFlow/Keras implementations utilized the entire GPU memory.
