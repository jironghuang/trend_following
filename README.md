#Developing a trend following model using futures

## Executive summary

Before June 2010, across different configuration of trend following strategies, trend following strategies perform well with sharpe of > 1, high calmar ratio, reasonable drawdowns.

After June 2010, risk adjusted returns, sharpe ratio dropped off signficantly with peformance dropping to below 0.7 for both strategies. 

Despite the deterioration in performance of trend following strategies, it may still have a place in an investor’s portfolio. According to Kathryn Kaminski, chief investment strategist at AlphaSimplex group and Visiting Lecturer in Finance at the MIT Sloan School of Management mentioned that trend following exhibits a crisis alpha characteristic. She studied 800 years of crisis and found that all crises create trends and there are opportunities for divergent strategies.

## Project motivation

As David Ricardo, a British economist in the 19th century once said, ‘cut short your losses and let your profits trend’ allude to the point that trend following as a profitable strategy could exist even back then.

Having read AQR’s papers on the Time Series Momentum (TSMOM, I am keen to explore this topic in the futures space (Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012)). Besides AQR papers, I have also followed closely the work of Robert Carver, an ex-MAN AHL quant who specialized in the space of intermediate to long term trend-following futures strategies. 

In this study, I will be exploring 2 approaches,

i.) TSMOM approach developed by AQR
ii.) Continuous forecasts approach (loosely based on Robert Carver’s framework in his books Leveraged Trading and Systematic Trading)

*Note: But because of limitations of the dataset that I will be using, I’m unable to incorporate ‘carry roll returns forecasts’ in Robert Carver’s books. I believe this would have impact on the effectiveness of the strategy.

The performance of strategies will be evaluated across 2 periods,

i.) In-sample period: 1984 to June 2010
ii.) Out-of-sample period: June 2010 to 2016

## Dataset

For this study, I will be using Futures dataset across 4 asset classes: Indices, Bonds, Currencies, Commodites provided by Quantopian (https://github.com/quantopian/research_public/blob/master/advanced_sample_analyses/TSMOM/TS%20MOM.ipynb) up till 2016. The continuous dataset is presumably stiched through backward, forward or proportional adjusted methodology (not explicitly mentioned in Quantopian’s github repository).


## How to use the repository

- I developed a futures research class to include the continuous forecast and Time Series Momentum (TSMOM) strategies.

- You may initialize the futures strategy class as follow,

#Initialize strategy class
strategy = FuturesResearch(data_path = "./quantopian_data/futures_incl_2016.csv", 
                           ewmac_variations = [8,16,32,64], 
                           breakout_variations = [40,80,160,320], 
                           optimize_weights_path = './optimize_weights', 
                           forecast_diff_before_rebal = 6.0,
                           notion_capital_per_position = 20000,
                           fix_capital = 500000,
                           commission = 20,
                           boostrap_sample_size = 300,
                           num_samples_per_period = 25,
                           prop_block_boostrap = 0.25,
                           max_annual_volatility = 0.15,
                           ind_instr_ref_volatility = 0.4                         
                           )

- Pls look at the jupyter notebook (futures_research.ipynb) to understand how to use the class
- Pls look at the write-up (Trend following strategy in futures using Time Series Momentum (TSMOM) and Continuous forecasts (CF).pdf) for the motivation of this study.
- Pls look at the documentation on the futures research class (futures_research_class_documentation.pdf)

Note that this is not an engineering project but more of a research project. Error handling and engineering features are not included. Should any of the repository for deployment purpose, these should be included. 



