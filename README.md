# AG-CNN Method
------
![method](https://github.com/Ien001/AG-CNN/blob/master/Screen%20Shot%202019-04-03%20at%2011.45.38%20AM.png)

## parameters

```json
{
	"NUM_EPOCH" : 50,
	"dataset" : {
		"resize" : [256, 256],
		"crop" : [224, 224]
	},
	"batch_size" : {
		"global" : 128,
		"local" : 64,
		"fusion" : 64
	},
	"optimizer" : {
		"SGD" : {
			"lr" : 0.01,
			"momentum" : 0.9,
			"weight_decay" : 0.0001
		}
	},
	"lr_scheduler" : {
		"step_size" : 10, 
		"gamma" : 0.1
	},
	"threshold" : 0.7,
	"backbone" : "resnet50"
}
```
## Result
------
![result best epoch model](https://github.com/maftuhm/lung-desease-detection-agcnn/blob/main/screenshot/exp9_best_epoch_model_result.png)
![result last epoch model](https://github.com/maftuhm/lung-desease-detection-agcnn/blob/main/screenshot/exp9_last_epoch_model_result.png)
