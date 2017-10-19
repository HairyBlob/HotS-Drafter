HotS Drafter

A neural network winrate estimator for the MOBA game Heroes of the Storm. The goal is to leverage a Monte-Carlo tree search approach to use this estimator as a drafting tool for competitive Heroes of the Storm.

Prerequisites

This project uses tensorflow-gpu to train the neural network, and the library scikit-mcts for the Monte-Carlo Tree Search. I dabbled a bit in the scikit-mcts library for it to return the top 4 picks. 

Use

The projet_tensorflow file is used to train a model on the HotsLogs data, you can download the last 30 days of data here: https://www.hotslogs.com/Info/API . Once you have a model up and running, you can use projet_restore to use it. Performance should reach about 61% accuracy on the testing set. Some example uses are: estimating winrate for a specific match, using MCTS to give the top picks in a given situation, or visualizing the network weights.

Authors

Daniel Gourdeau (HairyBlob#1378 on HotS)

