# G-Meta : Graph Meta Learning via Local Subgraphs - pyg version


### Introduction  
Implementation of **Graph Meta Learning via Local Subgraphs**.  
  
A **pyg version** implementation of "[Graph Meta Learning via Local Subgraphs](https://arxiv.org/abs/2006.07889)".
This code was created by referring to the [**official code**](https://github.com/mims-harvard/G-Meta), and it only deals with '**Single graph Disjoint labels**' problem. I hope it will help you with your research.

---

#### Authors: [Kexin Huang](https://www.kexinhuang.com), [Marinka Zitnik](https://zitniklab.hms.harvard.edu)

#### [Project Website](https://zitniklab.hms.harvard.edu/projects/G-Meta)

## Run
```bash
# Single graph disjoint label, node classification (Amazon_clothing)
python main.py --dataset Amazon_clothing
# Single graph disjoint label, node classification (Amazon_eletronics)
python main.py --dataset Amazon_eletronics
# Single graph disjoint label, node classification (dblp)
python main.py --dataset dblp
```

It also supports various parameters input:

```bash
python train.py --device # select a gpu device
                --dataset # 'Amazon_clothing', 'Amazon_eletronics', 'dblp'
                --way # num of way
                --shot # num of shot
                --qry # num of qry
                --epochs # epoch size
                --seed # seed number
                --num_seed # iter seed
                --episodes # num episodes per batch
                --patience # early-stopping patience 
                --h # 1 or 2 or 3: use h-hops neighbor as the subgraph.
                --hidden_dim # int: hidden dim size of GNN
                --meta_lr # float: outer loop learning rate
                --update_lr # float: inner loop learning rate
                --update_step # int: inner loop update steps during training
                --update_step_test # int: inner loop update steps during finetuning
```

## Cite Us

```
@article{g-meta,
  title={Graph Meta Learning via Local Subgraphs},
  author={Huang, Kexin and Zitnik, Marinka},
  journal={NeurIPS},
  year={2020}
}
```

## Codes borrowed from  
- [**G-Meta**](https://github.com/mims-harvard/G-Meta)
- [**GPN**](https://github.com/kaize0409/GPN_Graph-Few-shot)