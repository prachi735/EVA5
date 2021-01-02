# CODE 1
## Target: basic model with setup	
1. get test train data
2. set transforms
3. set data loaders
4. get data satistics
5. setup a simple model without any fancy stuff
6. set training & test loop
7. plot model result: accuracy & loss
## Result:
Parameters: 7,690	
Best Train Accuracy: 99.95%	
Best Test Accuracy: 99.22%	
## Analysis: 
1. The model performs well but can further be improved by training the model harder

# CODE 2
## Target: Add GAP and regularization
1. Added GAP in final layer
2. Added Batch Norm after every convolution
2. Increased the layes in model to boost it's capacity
## Result:
Parameters: 12,061
Best Train Accuracy: 99.54%
Best Test Accuracy: 99.71%
## Analysis:
1. The training and test accuracy have incresed 
2. The number of parameters is too high

# CODE 3 
## Target: 	Reduce the number of parameters
1. Reduced the number of kernels.
2. Added extra layer to compensate for lesser numner of kernels
## Result:
Parameters: 7,690	
Best Train Accuracy: 99.29%
Best Test Accuracy: 99.55%
## Analysis:
1. model performs good from the 12th epoch
2. the accuracy has decreased due to lesser number of parameters

# CODE 4
## Target: Improve the accuracy
Added more lalyers and increased the number of kernels
## Result:
Parameters: 8,610
Best Train Accuracy: 99.38%
Best Test Accuracy: 99.57%
## Analysis: 
The model has slightly better accuracy now as we increased the capacity of the model.

### Final Model layer: Input, Output, Receptive Field
                    
Layer|	N in|	p|	k|	s|	2p|	n out|	j|	r|	Output Shape|	Param #|
-----|	-----|	-----|	-----|	-----|	-----|	-----|	-----|	-----|	-----|	-----|
Conv2d-1|	28|	0|	3|	1|	|	28|	1|	|	[-1, 10, 26, 26]|	90|
Conv2d-4|	28|	0|	3|	1|	0|	26|	1|	3|	[-1, 10, 24, 24]|	900|
Conv2d-7|	26|	0|	3|	1|	0|	24|	1|	5|	[-1, 16, 22, 22]|	1440|
Conv2d-10|	24|	0|	1|	1|	0|	24|	1|	5|	[-1, 10, 22, 22]|	160|
MaxPool2d-13|	24|	0|	2|	2|	0|	12|	2|	6|	[-1, 10, 11, 11]|	0|
Conv2d-14|	12|	0|	3|	1|	0|	10|	2|	10|	[-1, 10, 9, 9]|	900|
Conv2d-17|	10|	0|	3|	1|	0|	8|	2|	14|	[-1, 16, 7, 7]|	1440|
Conv2d-20|	8|	0|	3|	1|	0|	6|	2|	18|	[-1, 16, 5, 5]|	2304|
Conv2d-23|	6|	0|	3|	1|	0|	4|	2|	22|	[-1, 10, 5, 5]|	160|
Conv2d-26|	4|	0|	3|	1|	0|	2|	2|	26|	[-1, 10, 3, 3]|	900|
AdaptiveMaxPool2d-29|	2|	0|	2|	1|	0|	1|	2|	28|	[-1, 10, 1, 1]|	0|
Conv2d-30|	1|	0|	1|	1|	0|	1|	2|	28|	[-1, 10, 1, 1]|	100|

Group:
1. Prachi Singh
2. Naman Shrimali 
3. M Enamulla
4. Amit Kumar Dubey
