# </br> Part 1: MSE Autoencoder

### </br> a. describe your dataset and the steps that you used to create it
### </br> b. provide a summary of your architecture 
### </br> c. discuss and explain your design choices,
### </br> d. list hyper-parameters used in the model,
### </br> e. plot learning curves for training and validation loss as a function of training epochs,
</br>
  <p align="center">
    <img src="/results/WithAugPlot.png" alt="Alt text" width="500"/>
  </p>
  
### </br> f. provide the final average error of your autoencoder on your test set

      Final average MSE error(testset): 0.0168

</br>
  <p align="center">
    <img src="/results/MSE_Result.png" alt="Alt text" width="500"/>
  </p>

### </br> g. provide a side-by-side example of 5 input and output images

</br>
  <p align="center">
    <img src="/results/autoencoder_results.png" alt="Alt text" width="500"/>
  </p>

### </br> h. discuss any decisions or observations that you find relevant.


# </br> Part 2: Encoder-Classifier Model

### </br> a. describe how you separated your dataset into classes

### </br> b. describe your classification technique and hyper-parameters

### </br> c. plot learning curves for training and validation loss for MSE and classification accuracy

</br>
  <p align="center">
    <img src="/results2/finalPlot.png" alt="Alt text" width="500"/>
  </p>

### </br> d. provide a side-by-side example of 5 input and output images

</br>
  <p align="center">
    <img src="/results2/finalResults.png" alt="Alt text" width="500"/>
  </p>

### </br> e. discuss how incorporating classification as an auxiliary tasks impacts the performance of your autoencoder

</br>
  <p align="center">
    <img src="/results2/multitaskResult.png" alt="Alt text" width="500"/>
  </p>

### </br> f. speculate why performance changed and recommend (but do not implement) an experiment to confirm or reject your speculation.



# </br> Part 3: Composite Image generator

### </br> a. specify which attribute you selected, the vector arithmetic applied and the resulting image(s),
### </br> b. provide a qualitative evaluation of your composite image, and


</br>
  <p align="center">
    <img src="/results3/latent_vector_arithmetic1.png" alt="Alt text" width="500"/>
  </p>

</br>
  <p align="center">
    <img src="/results3/attribute_composition1.png" alt="Alt text" width="500"/>
  </p>

### </br> c. discuss ways to improve the quality of your generated image.
