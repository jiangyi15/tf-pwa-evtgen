libtensorflow interface
=======================

Tensorflow provide [C API](https://tensorflow.google.cn/install/lang_c). (The official builds do not support slc6.)

And there is an cpp version [cppflow](https://github.com/serizba/cppflow) with simple interface  (require c++17).

TFPWA support to extract the amplitude model into tensorflow [SavedModel format](https://tensorflow.google.cn/guide/saved_model?hl=en).

```
config = ConfigLoader("config.yml")
config.save_tensorflow_model("model2")
```

Then, the `model2` will include `saved_model.pb` for the calculation,`variables` for the parameters.

The `main.cpp` include the base way to eval the amplitude from the model using [cppflow](https://github.com/serizba/cppflow).
The number of `p_i` is the number of final particles.

`build.sh` can be used to build it.
```
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.14.0.tar.gz
tar xvf libtensorflow-cpu-linux-x86_64-2.14.0.tar.gz

git clone https://github.com/serizba/cppflow.git

./build.sh
```
