# Model 39
## conv(2/3/4/5/6/7x32)[l2(0.01)] -> relu -> maxpool(2) -> dropout(0.8) -> batchnorm -> lstm(150) -> merge(concat) -> dropout(0.8) -> dense(1)  [32 batch size]
![diagram](https://github.com/ayenter/imdb_mud/blob/master/model_39/m39_diagram.png)
![graph](https://github.com/ayenter/imdb_mud/blob/master/model_39/m39_r1_e10_graph.png)