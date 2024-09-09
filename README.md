
This project focuses on analyzing the dynamics of digital conversations across various social platforms, including Reddit, Voat, Gab, and Facebook, with the aim of understanding how these platforms influence user interactions and participation in discussions. The research involves collecting data on conversations about topics such as politics, science, vaccines, and news to identify whether there are common characteristics or significant differences between the platforms.

A key aspect of the analysis is to understand how individual user behaviors (such as the time they join a conversation, the number of comments they post, and the time between comments) affect the collective dynamics of discussions. By using a statistical mechanics model, the study generates synthetic data that simulates these behaviors and explains how differences in collective conversation dynamics can be traced back to simple variations in individual user habits that depend on the platform.

The study concludes that the structure and algorithms of each platform play a significant role in shaping the form and lifespan of conversations. Platforms like Reddit foster more extended and distributed discussions, while others, like Facebook, see shorter, less interactive exchanges. These findings highlight that platform-dependent behaviors have a substantial impact on online discourse.

Repository Structure
The thesis project is accompanied by a GitHub repository that provides the code and resources used for data processing and analysis. The repository is organized into two main directories:

src: This folder contains all the core scripts and modules for data processing and analysis.
PRO: Manages the data preprocessing stages, filtering raw data to focus on the first 100 hours of conversations and threads with more than 50 comments.
EDA: Holds Jupyter notebooks for exploratory data analysis, including visualizations and plots.
SYN: Contains scripts for modeling user behavior and generating synthetic data, as well as the evaluation of the results.
docs: This folder includes documents and papers that inspired or were referenced during the project.
The repository is designed for transparency and reproducibility, allowing other researchers to follow the workflow from data preprocessing to synthetic data generation and analysis​(Conversation_dynamics_i…).
