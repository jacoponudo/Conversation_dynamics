# thesis
In this repository you will find all the material concerning theresearch for my  master thesis.
In the field of computational social sciences, this analysis is about conversation's dynamics and human behavior in digital environments.
 
The main finding from the data analysis are:
- dilatation of IAT is present by the final stages of the interaction, and are more frequent if the activity around is terminated[1]. 
- the toxicity's temperature of a thread, in the temporal window before a commment is conditioning the comment toxiciy[2].
- during a conversation the burst of activity can cause an increase of toxicity[3].
 

The scructure of the code consist in: 
- PRO: to extract new features from the dataset, and perform processing  of the row parquet file.  
- EDA: to do an exploratory data analysis of the dataset. 
- HWK: to fit an Hawkes process of point to the time series of comments for each user.
- TRA: to evaluate how toxicity is reverbering and affecting the comment's toxicity production.
- BUR: to analyze if there are some burst among conversation, or among user activity.


[1] An interaction is defined as the comments posted by a uset under a specific thread. To mesure the activity I've used a 1h window. This will be analyzed in HWK.
[2] To be decided how long the window should be, and how to quantify the propagation. Is the toxicity influencing in the same way even if the number of producers is more concentrated. This is in TRA.
[3] This is from Persistent Paper, will be in BUR folder.




Focus on : 
Attention of user with respect to a chat. 
