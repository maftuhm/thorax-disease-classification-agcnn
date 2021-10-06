# AG-CNN Method
------

## project still under development

![method](https://github.com/Ien001/AG-CNN/blob/master/Screen%20Shot%202019-04-03%20at%2011.45.38%20AM.png)

## main config parameters

```json
{
	"NUM_EPOCH" : 50,
	"num_classes" : 14,
	"dataset" : {
		"resize" : [256, 256],
		"crop" : [224, 224]
	},
	"batch_size" : {
		"global" : 120,
		"local" : 120,
		"fusion" : 120
	},
	"optimizer" : {
		"SGD" : {
			"lr" : 0.01,
			"momentum" : 0.9,
			"weight_decay" : 0.0001
		},
		"Adam" : {
			"lr": 0.001,
			"betas": [0.9, 0.999], 
			"eps": 1e-08, 
			"weight_decay" : 0.0001
		}
	},
	"lr_scheduler" : {
		"StepLR" : {
			"step_size" : 20, 
			"gamma" : 0.1,
			"verbose" : true
		},
		"ReduceLROnPlateau" : {
			"mode" : "min", 
			"patience" : 3,
			"factor" : 0.1,
			"verbose" : true
		}
	},
	"max_batch" : {
		"global" : 16,
		"local" : 14,
		"fusion" : 4
	}
}
```
