# Pipeline for "Graph SSL Scaling"

## Instructions

## How to set up and check the results with this pipeline?

1. The **ConfigParser** will read the config file of your settings(e.g., learning rate,hidden_size) or default settings and set them for datasets or the corresponding executor of the model chosed by you.

2. The **./libgptb/log folder** will restore the config settings and the **./libgptb/cache folder** will restore the evaluate results with the exam_id. Then you can easily collect them.


## How to use the pipeline?

You can find the whole process in `libgptb/pipeline/pipeline.py`

In brief the whole process will be like

1. **ConfigParser** will load the config file and default config file for the **Dataset** and **Executor**
2. **Dataset** will load the data and get some features(e.g., input feature dims)
3. **Executor** will load the chosen model and the model will be trained and evaluated with stored evaluation results.


### Requirements.txt

```
torch_geometric 
tqdm
numpy
scikit-learn
networkx
PyGCL
```

### Usage

```shell
python3 ./run_model.py --task SSGCL --model InfoGraph --dataset ogbg-molhiv --ratio 0.1 --config_file random_config/config_1
```



