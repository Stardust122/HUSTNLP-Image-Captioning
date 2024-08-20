# HUSTNLP-Image-Captioning

This is the repository of the final project "Image Description Generation and Multimodal Learning" of NLP. This repository contains 2 parts of our work: image description generation using multimodal large language models(prompt engineering), image captioning using CNN-RNN and CNN-RNN+Attention.

The original repository of used CNN-RNN is [here](https://github.com/ksheersaagr/Automatic-Image-Captioning), and the original repository of used CNN-RNN+Attention is [here](https://github.com/ruotianluo/ImageCaptioning.pytorch).

## Pretraind models

We provide the pretrained models of CNN-RNN and CNN-RNN+Attention [here](https://pan.baidu.com/s/1T7tPXeh-cFG9N7Jbw9Q_Hg?pwd=p0z9).

## Usage of CNN-RNN

### Installation

- Required packages:
  
```
torch
torchvision
numpy
nltk
matplotlib
```

### Training

- Prepare the dataset in COCO format
- If you have the corresponding vocab.pkl, then you can directly use it, otherwise you need to generate the vocab.pkl file by changing "vocab_from_file" to False in [train.py](CNN-RNN/train.py)
- Change the path in [train.py](CNN-RNN/train.py) and data_loader.py to your local path, containing the dataset, the vocab.pkl file and the model path
- Change the hyperparameters in [train.py](CNN-RNN/train.py) if necessary and run the following command:
  
```
cd CNN-RNN
python train.py
```

### Inference

If you want to infer the total test set and generate the result file, run the following command:

```
cd CNN-RNN
python inference.py
```

If you want to infer several images, Uncomment "get_prediction()" in inference.py line 108 and comment the line below it, then run the command above.

- To compute the metrics, change the path in metrics.py and run the following command:

```
cd CNN-RNN
python metrics.py
```

## Usage of CNN-RNN+Attention

## Requirements

- Python 3
- PyTorch 1.3+ (along with torchvision)
- [cider](https://github.com/vrama91/cider) 
- [coco-caption](https://github.com/tylin/coco-caption)(**Remember to follow initialization steps in coco-caption/README.md**)
- yacs
- lmdbdict
- Optional: pytorch-lightning

## Install

If you have difficulty running the training scripts in `tools`. You can try
```
cd CNN-RNN+Attention
python -m pip install -e .
```

## Train

### Prepare data

See details in [CNN-RNN+Attention/data/README.md](CNN-RNN+Attention/data/README.md). We alse provide the preprocessed data [here](https://pan.baidu.com/s/1lkhSj06TpcUUNPNnzG_pyQ?pwd=p0ag) for the dataset of NLP course final project.

### Start training

```
cd CNN-RNN+Attention
python tools/train.py --id fc --caption_model newfc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30
```

or

```
cd CNN-RNN+Attention
python tools/train.py --cfg configs/fc.yml --id fc
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `log_$id/`). By default only save the best-performing checkpoint on validation and the latest checkpoint to save disk space. You can also set `--save_history_ckpt` to 1 to save every checkpoint.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

To checkout the training curve or validation curve, you can use tensorboard. The loss histories are automatically dumped into `--checkpoint_path`.

The current command use scheduled sampling, you can also set `--scheduled_sampling_start` to -1 to turn off scheduled sampling.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to pull the submodule `coco-caption`.

For all the arguments, you can specify them in a yaml file and use `--cfg` to use the configurations in that yaml file. The configurations in command line will overwrite cfg file if there are conflicts.  

### Generate image captions

Now place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

```
cd CNN-RNN+Attention
python tools/eval.py --model model.pth --infos_path infos.pkl --image_folder blah --num_images 10
```

This tells the `eval` script to run up to 10 images from the given folder. If you have a big GPU you can speed up the evaluation by increasing `batch_size`. Use `--num_images -1` to process all images. The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```
cd CNN-RNN+Attention/vis
python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser and you should see your predicted captions.

## Statement

The code in this repository is only used for academic research and personal practice. If you find this work or code useful for your research, please cite the original papers and repositories of the authors. If you have any questions, please feel free to contact me.
