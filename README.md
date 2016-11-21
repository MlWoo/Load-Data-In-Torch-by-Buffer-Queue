# Load-Data-In-Torch-by-Buffer-Queue
The project aims at loading data more efficiently by using buffer queue in Torch DL framework. Although LMDB is used as dataset source, the tricks which boost DNN training in the project are pratically effective if you prefer to raw images.

Reading data and training network are two main parts of deep learning procedures. It's reasonable to allocate one thread to fetch data because reading data is light-weight work and accessing records orderly is necesary . If the two parts are executed serially, it means computing resources waste because reading data occupies only one thread but all other threads are idle when it runs. 

It's well known that computation in deep network costs much more time than reading data from raw data or database. And the former will use data which is provided by the later. So producer-consumer is an appropriate manner to arrange the two parts.

#Requirements
1. nnlr(https://github.com/gpleiss/nnlr)
2. LMDB-torch(https://github.com/eladhoffer/lmdb.torch)
3. tunnel(https://github.com/zhangxiangxiao/tunnel)

#How to create LMDB
Using the CreateLMDBs.lua(https://github.com/eladhoffer/ImageNet-Training).

#How to allocate openmp threads to 2 torch Threads(more similar to Process)
There are 44 cores on my machine, so there are 44 available threads to be allocated to 2 torch threads. The amount of threads using by DNN training procedure should be even and less than the total available threads(44 on my machine). The more threads you allocate to it, the better performance the machine will achieve. So 42 threads is a good choice for DNN training part, the remained threads (here is 2 on my machine) to read dataset. The experience above can be viewed as a recommendation. You'd better tune your application to get the best performance on your machine.

There's an another trick to accelerate your application especially when the amount of batch is small such as googlenet. Reading a batch of images is much faster than training DNN using the images. The former costs about 50ms, but the later costs about 300ms on my machine when training googlenet model. So reading thread is always waiting for computing thread to fetch data in the buffer when co-work of them is stable. So it's wonderful to allocate all cores to computing threads when reading thread is suspended. The larger the buffer queue gets, the longer time computing thread which utilizes all cores occupies.  

1. Allocate shareing buffer as large as possible according to your RAM capacity. 
2. When the buffer is filled with unused data, reading thread will be suspended, allocate all cores to computing threads.
3. When fresh data is nearly empty, Resume the reading thread, and reset 42 sores to computing thread to avoid undesired thread-race.


#Explanation
config.lua and opts.lua are used to configure parameters of your model or data.

LMDBProvider.lua provides interfaces to access LMDB and wrapps the data and method into a torch class.

model.lua provides interfaces to create the model which includes network and optimizer, and wrapps the data and method into a torch class.

main.lua has a clear procedure to train network. Producer and consumer threads are coroutine by judging the vector (buffer queue) is full or empty.

  

