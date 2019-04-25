# Project 4: Decision Trees for Expert Iteration (Reinforcement learning)
This is the project I have been assigned to, description of which can be found in [CE888_Project_4](CE888_Project_4.pdf) file.

The initial training [data](data.csv) have been generated using the [python script](UCT_initial.py). This is a slightly modified version of [this](http://mcts.ai/code/python.html) code. The data have been generated for the OXO game (a.k.a. Tic-Tac-Toe) using Monte Carlo Tree Search with greedy Upper Confidence Bounds policy. To counteract potentially detrimental correlations between the states generated, only a single state from each game was selected. This was repeated for 10,000 games, which resulted in 10,000 states collected (see data.csv file).

## Assignment 1
- Code used: UCT_initial.py
- Data generated: data.csv

## Assignment 2
- Code used: UCT_DT.py
- Initial data.csv enhanced into data_e.csv via enhance_data.py (swapping players)
- Folder "experiments" contains all experimental results
- Folder "final_results" contains only those experimental results used in the final report
- Folder "images" contains all the plots produced for the report purposes
- "Analyse results" and "Knowledge extraction" notebooks were used to analyse experiments and produce plots