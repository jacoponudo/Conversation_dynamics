# thesis
In this repository you will find all the material concerning theresearch for my  master thesis.
In the field of computational social sciences, this analysis is about conversation's dynamics and human behavior in digital environments.
 
Starting from some signals that caracterize the plattform related macro dynamics of conversation, I've tried to replicate synthetically data  from each platform to fit these signals, and then be able to interpret the different combinations of parameters used.

As  signals I've identified the tendency of the conversation of been composed by less first comments with time; and also the tendency of the conversation of be consumed more repidly on some platforms than on some others. 

The first model to replicate  synthetically the data is composed by: 
- sample of number of users from a scale free (plaftorm specific)
- sample the moment of entrance in a conversation of a use, using a beta (plaform specific)
- sample from the mixture of an exponenital zero inflated the number of comment that a user will do (plaform specific)
- dispose these comments in the time using the IAT suggested by specific borr distribution (plaform specific)

Then compute the differences and quantify the loss to evalutate the fitting, and finally give a logical interpretetion to the  best set of parameters.

In order to go from eaw data to processed, there is the directory PRO, and using PRO_mainclass.ipynb is possible to perform the preproceesing.
In order to plot most of the graphs related to the explotatory data analysis there is the directory EDA.
In order to estimate sets of paramters for each platform, replicate synthetic data and compare those with real data there is the directory SYN.

Specifically in SYN there will  be: 

SYN_mainclass.ipynb
- Paramters estimation
- Production of synthetic data
- Comparison of observed and simulated data
  +
SYN_interpretation.ipynb
-intrpretate parameters for each platform 
