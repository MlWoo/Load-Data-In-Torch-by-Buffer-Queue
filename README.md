# Load-Data-In-Torch-by-Buffer-Queue
Using buffer queue to load data  more efficiently torch

Reading data and training network are two main parts of deep learning procedures. It's reasonable to allocate one thread to fetch data because reading data is light-weight work and accessing records orderly is necesary . If the two parts are executed serially, it means computing resources waste because reading data occupies only one thread but all other threads are idle when it runs. 

It's well known that computation in deep network costs much more time than reading data from raw data or database. And the former will use data which is provided by the later. So producer-consumer is an appropriate manner to arrange the two parts.

#Requiments
1. nnlr(https://github.com/gpleiss/nnlr)
2. LMDB-torch(https://github.com/eladhoffer/lmdb.torch)
3. tunnel(https://github.com/zhangxiangxiao/tunnel)

#Explanation
config.lua and opts.lua are used to configure parameters of your model or data.

LMDBProvider.lua provides interfaces to access LMDB and wrap the data and method into a torch class.

model.lua provides interfaces to create the model which includes network and optimizer, and wrap the data and method into a torch class.

main.lua has a clear procedure to train network. Producer and consumer threads are coroutine by judging the vector (buffer queue) is full or empty.

  

