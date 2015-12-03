# sortable-matching

This is a script to find matches between products and product descriptions. Using a bag-of-words approach, I first calculate the tf-idf weights for each token in the bag. Treating the weights as components of a vector allows the cosine similarity to be calculated. The threshold parameter is somewhat arbitrary, but could be tuned with a comparison to a gold standard. 
