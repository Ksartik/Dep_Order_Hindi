# Dep_Order_Hindi
Determining distance and order of constitutents from psycholinguistic factors

# Data 

1. The main corpus data is taken from [here](http://www.cfilt.iitb.ac.in/iitb_parallel/). This is used to calculate the HDMI values and do the further analysis with distance and word-order. 

2. To get the animacy annotations of the nouns, we extracted the annotated nouns collected in the data used in this [work](https://www.aclweb.org/anthology/W13-2320/). 

3. We use the pre-trained word2vec models to calculate semantic similarity. These are available [here](https://www.aclweb.org/anthology/W13-2320/).

# Code organization 

## Parsing 

The parser used in this work is taken from [here](https://bitbucket.org/iscnlp/workspace/projects/ISCNLP). **parse_data.py** parses the main corpus data using this parser and stores it in a more easy-to-work format. In addition, it adds an additional argument-structure information on the verbal POS tag as well. We will use this reformatted parsed data for all the following tasks.

## HDMI calculation 

**hdmi_online_pos.py** calculates the HDMI values for dependencies at a given distance. It calculates these values separately for different dependency relations (argument, adjunct, subject, direct object and indirect object) depending on whether the noun is case-marked or not. 

In order to calculate HDMI at multiple distances, this code can be run parallely using **hdmi_parallel.sh** at various distances we might want to consider (we consider 1-20 here). It will store these values in a file **hdmi_dist_df_pos.csv**

## Anlayzing the variation of dependency distance

**dist_param_analyze.py** lists the dependencies along with the values of the factors that will affect the the corresponding dependency distance. The calculation of these factors is according to the scheme mentioned in the paper and using the data mentioned above. HDMI values as calculated in the previous step will also be used here. 

**analysis_dist.R** then analyzes this data using regression models and assisting plots. 

## Analyzing the variation of word-order

**order_param_analyze.py** lists the pair of dependencies in a sentence along with the values of the factors that will affect the the corresponding order of the constituents in this pair. The calculation of these factors is according to the scheme mentioned in the paper and using the data mentioned above. HDMI values as calculated in the previous step will also be used here. 

**analysis_order.R** then analyzes this data using regression models and assisting plots. 
