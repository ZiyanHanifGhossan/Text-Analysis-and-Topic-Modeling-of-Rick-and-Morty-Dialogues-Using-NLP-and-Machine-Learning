# Text-Analysis-and-Topic-Modeling-of-Rick-and-Morty-Dialogues-Using-NLP-and-Machine-Learning
This project focuses on text analysis of the Rick and Morty television series scripts. The objective is to explore and understand dialogue patterns, identify recurring themes, and cluster dialogue lines based on their content using various Natural Language Processing (NLP) and Machine Learning techniques.

**Workflow Overview**

**Data Loading and Cleaning:**
The script data was loaded from a CSV file into a Pandas DataFrame, with the dialogue text extracted from the *line* column. Initial preprocessing included converting text to lowercase, removing numerical characters using regular expressions, and eliminating empty dialogue entries to ensure clean input data.

**Text Feature Extraction:**
Two text vectorization techniques were applied:

* **CountVectorizer:**
  Converted dialogue lines into a bag-of-words representation, where each document is represented by token counts. English stop words were removed to eliminate common words with limited semantic value.

* **TF-IDF Vectorizer:**
  Transformed dialogue text into weighted numerical features using Term Frequency–Inverse Document Frequency, highlighting words that are both frequent within a dialogue and distinctive across the corpus. Stop words were removed, and *min_df* and *max_df* parameters were applied to filter extremely rare and overly common terms.

**Clustering with K-Means:**
K-Means clustering was applied to TF-IDF features to group dialogue lines based on semantic similarity. The optimal number of clusters was determined using the Elbow Method by analyzing inertia values.

Cluster interpretation was performed by extracting the most influential words from each cluster centroid, allowing thematic labeling of dialogue groups.

**Dimensionality Reduction with Singular Value Decomposition (SVD):**
To reduce high dimensionality and noise in the TF-IDF feature space, SVD was applied to reduce features to 100 dimensions. K-Means clustering was then performed on the reduced feature set. Cluster centroids were mapped back to the original feature space using inverse transformation to extract representative keywords.

**Latent Dirichlet Allocation (LDA):**
LDA topic modeling was applied to CountVectorizer features to identify latent topics across the dialogue corpus. The model identified multiple abstract topics, each represented by a set of high-probability keywords, enabling thematic exploration of the series’ narrative structure.

**Results and Insights:**
K-Means clustering (both TF-IDF and SVD-based) successfully grouped dialogue lines according to semantic similarity, revealing clusters centered around characters, emotional expressions, and recurring actions. LDA uncovered broader thematic patterns and narrative elements present throughout the series. Overall, this project demonstrates how NLP and unsupervised learning techniques can be used to extract meaningful insights from large-scale dialogue data.
