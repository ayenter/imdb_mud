# Model 36
## conv(2/3/4/5/6/7x32)[l2(0.01)] -> relu -> maxpool(2) -> dropout(0.5) -> dense(24) -> dropout(0.5) -> batchnorm -> lstm(150) -> merge(concat) -> dense(1)  [32 batch size]
![diagram](https://github.com/ayenter/imdb_mud/blob/master/model_36/m36_diagram.png)
![graph](https://github.com/ayenter/imdb_mud/blob/master/model_36/m36_r1_e10_graph.png)