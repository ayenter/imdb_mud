# Model 47
## conv(2/3/4/5/6/7x512)[l2(0.01)] -> relu -> maxpool(32) -> dropout(0.7) -> batchnorm -> lstm(512) -> merge(concat) -> dropout(0.7) -> dense(1)  [ 32 batch size ]
![diagram](https://github.com/ayenter/imdb_mud/blob/master/model_47/m47_diagram.png)
![graph](https://github.com/ayenter/imdb_mud/blob/master/model_47/m47_r1_e10_graph.png)