# GraphSAINT-NRW, ERW
- Juyeong Shin, Young-Koo Lee
- KSC 2022 Paper
- link: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11224420

- Modified from DGL GraphSAINT example
    - Paper link: [https://arxiv.org/abs/1907.04931](https://arxiv.org/abs/1907.04931)
    - Author's code: [https://github.com/GraphSAINT/GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)
    - DGL example code: [https://github.com/dmlc/dgl/tree/master/examples](https://github.com/dmlc/dgl/tree/master/examples)

## Dependencies

- Python 3.10.12
- PyTorch 2.0.1
- NumPy 1.25.0
- Scikit-learn 1.2.2
- DGL 1.1.1

## Dataset

All datasets used are provided by Author's [code](https://github.com/GraphSAINT/GraphSAINT). They are available in [Google Drive](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [Baidu Wangpan (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg#list/path=%2F)).

## Config

- The config file is `config.py`, which contains best configs for experiments below.
- Please refer to `sampler.py` to see explanations of some key parameters.

### Parameters

| **aggr**                                                     | **arch**                                                     | **dataset**                                                  | **dropout**                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| define how to aggregate embeddings of each node and its neighbors' embeddings ,which can be 'concat', 'mean'. The neighbors' embeddings are generated based on GCN | e.g. '1-1-0', means there're three layers, the first and the second layer employ message passing on the graph, then aggregate the embeddings of each node  and its neighbors. The last layer only updates each node's embedding. The message passing  mechanism comes from GCN | the name of dataset, which can be 'ppi', 'flickr', 'reddit', 'yelp', 'amazon' | the dropout of model used in train_sampling.py               |
| **edge_budget**                                              | **gpu**                                                      | **length**                                                   | **log_dir**                                                  |
| the expected number of edges in each subgraph, which is specified in the paper | -1 means cpu, otherwise 'cuda:gpu', e.g. if gpu=0, use 'cuda:0' | the length of each random walk                               | the directory storing logs                                   |
| **lr**                                                       | **n_epochs**                                                 | **n_hidden**                                                 | **no_batch_norm**                                            |
| learning rate                                                | training epochs                                              | hidden dimension                                             | True if do NOT employ batch normalization in each layer      |
| **node_budget**                                              | **num_subg**                                                 | **num_roots**                                                | **sampler**                                                  |
| the expected number of nodes in each subgraph, which is specified in the paper | the expected number of pre_sampled subgraphs                 | the number of roots to generate random walks                 | specify which sampler to use, which can be 'node', 'edge', 'rw', corresponding to node, edge, random walk sampler |
| **use_val**                                                  | **val_every**                                                | **num_workers_sampler**                                      | **num_subg_sampler**                                            |
| True if use best model to test, which is stored by earlystop mechanism | validate per 'val_every' epochs                              | number of workers (processes) specified for internal dataloader in SAINTSampler, which is to pre-sample subgraphs | the maximal number of pre-sampled subgraphs                  |
| **batch_size_sampler**                                          | **num_workers**                                              |                                                              |                                                              |
| batch size of internal dataloader in SAINTSampler            | number of workers (processes) specified for external dataloader in train_sampling.py, which is to sample subgraphs in training phase |                                                              |                                                              |


## Minibatch training

Run with following:
```bash
python train_sampling.py --task $task $online
# online sampling: e.g. python train_sampling.py --task ppi_n --online
# offline sampling: e.g. python train_sampling.py --task flickr_e
```

- `$task` includes `ppi_n, ppi_e, ppi_rw, flickr_n, flickr_e, flickr_rw, reddit_n, reddit_e, reddit_rw, yelp_n, yelp_e, yelp_rw, amazon_n, amazon_e, amazon_rw`. For example, `ppi_n` represents running experiments on dataset `ppi` with `node sampler`
- If `$online` is `--online`,  we sample subgraphs on-the-fly in the training phase, while discarding pre-sampled subgraphs. If `$online` is empty, we utilize pre-sampled subgraphs in the training phase.
