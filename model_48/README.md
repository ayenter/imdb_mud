# Model 48
## conv(2/3/4/5/6/7x512)[l2(0.01)] -> relu -> maxpool(32) -> dropout(0.8) -> batchnorm -> lstm(512) -> dropout(0.8) -> merge(concat) -> dropout(0.7) -> dense(1)  [ 32 batch size ]
![diagram](https://github.com/ayenter/imdb_mud/blob/master/model_48/m48_diagram.png)
![graph](https://github.com/ayenter/imdb_mud/blob/master/model_48/m48_r1_e10_graph.png)