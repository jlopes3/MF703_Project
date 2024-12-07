# MF703_Project

## Abstract/Introduction
With the United States presidential elections becoming increasingly polarized, many citizens express concerns about the country's future. Gerardo Manzo, in his study, highlights how Italian financial markets respond to politically charged events This repository uses Manzo's analysis as motivation to analyze market volatility during United States election cycles and leverage this volatility to construct high-performing investment portfolios.


We begin by defining the election period: for this analysis, it includes the 26 weeks preceding an election and the 2 weeks following it. We then examine whether market volatility during these periods is significantly higher compared to similar timeframes in non-election years. Finally, we collect data on multiple assets and apply portfolio optimization techniques to explore potential performance improvements.

Accompanying paper will be pushed to the repository by December 8th, 2024

## Assumptions
For simplicity of our portfolios, we assume that there is an infinite budget. We discuss further that we assign specific weights to different assets based on constraints that we give our weight optimizer. We also assume zero transaction costs to the positions that we hold, because the main focus of this project is the weight optimization rather than the simulation of trades. In particular, we assume costless rolling of futures contracts.  We finally assume that log normality of returns for each asset class. More assumptions on the government bond data will be discussed in the Treasuries portion of the Financial Instruments section.
