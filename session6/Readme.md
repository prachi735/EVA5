This notebook implements below loss & regularization:

    1. L1 + BN
    2. L2 + BN
    3. L1 and L2 with BN
    4. GBN
    5. L1 and L2 with GBN

### Model performance
                    
Loss function	| Train Accuracy |	Loss	| Best Test Accuracy | Loss 
----- |-----|---|---|---
None*|	99.38%|	0.0354|	99.57%|0.0137
L1 with BN|	99.19%|	0.1481|	99.54%|0.0178
L2 with BN|	99.36%|	0.0089|	99.74%|0.0135
L1 + L2 with BN|	99.32%|	0.0573|	99.68%|0.0137
GBN|	99.29%|	0.0210|	99.69%|0.0144
L1 + L2 with GBN|	99.11%|	0.0748|	98.59%|0.0175

*None is the base model without any loss regularization or GBN. The code for the model can be found ![here](https://github.com/prachi735/EVA5/blob/main/Session5/EVA5_S5_F4.ipynb)*

### Accuracy Graph
![Accuracy Graph](https://github.com/prachi735/EVA5/blob/main/session6/Graphs/test_acc.png) 

### Loss Graph
![Train Loss Graph](https://github.com/prachi735/EVA5/blob/main/session6/Graphs/train_loss.png) 
![Test Loss Graph](https://github.com/prachi735/EVA5/blob/main/session6/Graphs/test_loss.png) 

### Misclassified Images with GBN Model
![Misclassified Images with GBN Model](https://github.com/prachi735/EVA5/blob/main/session6/Graphs/misclassified_images_with_gbn_model.png) 


## Conclusion:

Here we see L1 regularization has decreased the performance slightly but L2 regularization improves it as expected.
The decrease in L1 performance could be due to it's feature selection nature that sometimes prevents it from learning complex pattern.
L2 being more robust is able to learn better complex features, hence better test results.
GBN also boots model's accuracy and also speeds up the training of the model as seen in the graph.
