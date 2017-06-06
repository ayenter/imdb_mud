# Model 42
## conv(2/3/4/5/6/7x32)[l2(0.01)] -> relu -> maxpool(2) -> dropout(0.8) -> batchnorm -> lstm(150) -> merge(concat) -> dense(1)  [ 32 batch size | 10,000 top words | 1,000 word cap ]
![diagram](https://github.com/ayenter/imdb_mud/blob/master/model_42/m42_diagram.png)
![graph](https://github.com/ayenter/imdb_mud/blob/master/model_42/m42_r1_e10_graph.png)