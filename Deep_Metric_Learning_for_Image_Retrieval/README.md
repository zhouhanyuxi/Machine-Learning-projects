# Deep Metrics for Image Retrieval tasks
## Team StaySafe
2020.5.24 - 2020.8.18
### Introduction to Project
Metric learning endeavors to map data to an embedding space where similar data are close to each other, and dissimilar data are far from each other.  
This project aims to reveal and compare different metric learning algorithms’ performance by implementing an image retrieval web application that has a standard pipeline for multiple algorithms. There are two general pipelines in this project, one for training and the other for image retrieval. The web application of this project is built on a Flask web framework and Apache2 web server.  
As a result of the project, two algorithms, multi-similarity, and soft-triple appeared to over-perform under the same experimental setting.  

### My work
(1) Technical aspects:  
- Research:  
Complete background reading of the deep metric learning area and learn SoftTriple loss in detail.  
Write summaries of reading materials and communicate with other team members to discuss each other's reading materials.  
Then, reproduce the SoftTriple model according to relevant papers and codes. Adjust the SoftTriple model, train it with the CUB-200-2011 dataset, and generate the corresponding embedding space file as well as image path file.  
Finally, help Zhao Yuan to train the SoftTriple model with the Cars-196 dataset.
- Development:  
Set up our web server and web framework on Azure virtual machine prepared by Yikai Wang.  
Then, finish the controller part of the search page and the main back end part of the user rating page, and cooperate with Yili Lai to make sure the front and back end work together.  
Help with model deployment and debug.  
Finally, recreate our virtual machine and reset up our web server as well as application in a new Azure subscription after the balance of the original subscription ran out.

(2) Non-technical part:
Complete the implementation part of our final report.  
Finish most of the minutes of daily group meetings.  

### Introduction to SoftTriple
SoftTriple loss learns the embeddings directly from deep neural networks and without triplet constraints sampling.  
Qi Qian et al. found SoftMax loss is equivalent to a smoothed triplet loss where each class has a single center and proposed SoftTriple loss, which extends the SoftMax loss with multiple centers for each class.  
So SoftTriple loss is demonstrated on the finegrained visual categorization tasks which optimize the geometry of local clusters, provides more flexibility for modeling intra-class variance like triplet constraints, but reduces the total number of triplets to alleviate the challenge from a large number of triplets like using proxies.  
Moreover, optimizing SoftTriple loss can learn the embedding without the sampling phase by mildly increasing the size of the last fully connected layer, since most samples cannot represent data well and would lead  to sub-optimal embedding.


