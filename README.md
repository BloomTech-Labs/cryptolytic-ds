# Cryptolytic

You can find the project at [Cryptolytic](http://www.cryptolyticapp.com/).

## Contributers
### Team Lead
 |                                       [Stanley Kusmier](https://github.com/standroidbeta)                                        |
| :-----------------------------------------------------------------------------------------------------------: |
|                      [<img src="https://github.com/alqu7095/Cryptolytic_README/blob/master/Stan.png?raw=true" width = "200" />](https://github.com/standroidbeta)                       |
|                 [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/standroidbeta)                 |
| [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/stanley-kusmier-0563a153/) |
### Data Science


|                                       [Alfredo Quintana](https://github.com/alqu7095)                                        |                                       [Elizabeth Ter Sahakyan](https://www.github.com/elizabethts)                                        |                                       [Marvin A Davila](https://github.com/MAL3X-01)                                        |                                       [Nathan Van Wyck](https://github.com/nrvanwyck)                                        |                                       [Taylor Bickell](https://github.com/tcbic)                                        |
| :-----------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
|                      [<img src="https://github.com/alqu7095/Cryptolytic_README/blob/master/Alfredo.jpeg?raw=true" width = "200" />](https://github.com/alqu7095)                       |                      [<img src="https://github.com/alqu7095/Cryptolytic_README/blob/master/Elizabeth.jpeg?raw=true" width = "200" />](https://www.github.com/elizabethts)                       |                      [<img src="https://github.com/alqu7095/Cryptolytic_README/blob/master/Marvin.jpeg?raw=true" width = "200" />](https://github.com/MAL3X-01)                       |                      [<img src="https://github.com/alqu7095/Cryptolytic_README/blob/master/Nathan.png?raw=true" width = "200" />](https://github.com/nrvanwyck)                       |                      [<img src="https://github.com/alqu7095/Cryptolytic_README/blob/master/Taylor.jpeg?raw=true" width = "200" />](https://github.com/tcbic)                       |
|                 [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/alqu7095)                 |            [<img src="https://github.com/favicon.ico" width="15"> ](https://www.github.com/elizabethts)             |           [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/MAL3X-01)            |          [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/nrvanwyck)           |            [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/tcbic)             |
| [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/alfredo-quintana-98248a76/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/elizabethts) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/marvin-davila/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/nathan-van-wyck-48586718a/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/taylorbickell/) |






## Project Overview


[Trello Board](https://trello.com/invite/b/HY6gCjh2/3f8eb169fd2f1e2415abf535c20accb3/labs17-cryptolytic)

[Product Canvas](https://www.notion.so/e563b27ab8e94ce2a3f7b536fc365715?v=3781e3eb9e72447f9262ebacd1e21fa9)


Cryptolytic is a platform for beginners tinkering with cryptocurrency to the seasoned trader. It provides you with recommendations on when to buy and sell based on technical indicators and assesses the markets to predict arbitrage opportunities before they even happen.


### Figma Prototype 
<img src="https://github.com/alqu7095/Cryptolytic_README/blob/master/cryptolytic_thumbnail.png?raw=true" width = "500" />

### Tech Stack

Python, AWS, PostgreSQL, SQL, Flask

### Predictions

Each trade recommender model recommends trades for a particular trading pair on a particular exchange by predicting whether the closing price will increase by enough to cover the costs of executing a trade.

The arbitrage models predict arbitrage opportunities between two exchanges for a particular trading pair.  Predictions are made five minutes in advance. To count as an arbitrage opportunity, a price disparity between two exchanges must last for at least thirty minutes, and the disparity must be great enough to cover the costs of buying on one exchange and selling on the other.

The trained and pickled models can be accessed via the organization's AWS S3 bucket by the name of "crypto-buckit" under folder aws/models. Current code (as of 4 February 2020) will upload all future models into this S3 bucket.

The naming convetion for the models is model\_{arbitrage/trade}\_{api}\_{trading\_pair}.pkl

The predictions themselves can be accessed via the Organization's AWS RDS database with the table name "predictions".

### Features

Each of the nine trade recommender models is trained on 80 features.  Of those 80 features, five are taken directly from the OHLCV data (open, high, low, close, base\_volume), and the remainder are technical analysis features. We are filling NaN values of open, high, low, close with the average price, and forward filling NaN values for base\_volume

Each of the arbitrage models is trained on 80 features.  Of those 80 features, and four indicate the degree and length of price disparities between two exchanges (higher_closing_price, pct_higher, arbitrage_opportunity, window_length).  Arbitrage is calculated by comparing the price of the primary exchange against the mean price of the other exchanges. This allows us to compare the market against every other market with minimal computation cost.

Technical analysis features were engineered with the Technical Analysis Library; they fall into five types:
(1) Momentum indicators
(2) Volume indicators
(3) Volatility indicators
(4) Trend indicators
(5) Others indicators

Documentation for the technical analysis features features is available here:

[Technical Analysis Library Documentation](https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html)

### Data Sources

We obtained all of our data from the Cryptowatch, Bitfinex, Coinbase Pro, and HitBTC APIs. Documentation for obtaining that data is listed below:

[Cryptowatch REST API OHLCV Data Documentation](https://docs.cryptowat.ch/rest-api/)

[Bitfinex API OHLCV Data Documentation](https://docs.bitfinex.com/reference#rest-public-candles)

[Coinbase Pro API OHLCV Data Documentation](https://docs.pro.coinbase.com/?r=1#get-historic-rates)

[HitBTC OHLCV Data Documentation](https://api.hitbtc.com/#candles)

[Binance API OHLCV Documentation](https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md)

[Gemini REST API OHLCV Data Documentation](https://docs.gemini.com/rest-api/)

[Kraken REST API OHLCV Data Documentation](https://www.kraken.com/en-us/features/api)

[Poloniex API OHLCV Data Documentation](https://docs.poloniex.com/#introduction)

### Python Notebooks

[Notebook Folder](https://github.com/Lambda-School-Labs/cryptolytic-ds/tree/master/finalized_notebooks)


## How to connect to the Cryptolytic API
 http://www.cryptolyticapp.com/

### Trade API [/trade](http://www.cryptolyticapp.com/trade)

 Method: ["GET"]

 Returns: ``` {"results":
"{('exchange', 'trading_pair'): [{
'p_time': 'time',
‘period’: ‘minutes’,
'prediction': 'result'}], }"} ```
  
### Arbitrage API [/arbitrage](http://www.cryptolyticapp.com/arbitrage)
  
Method: ["GET"]

Returns: ``` {"results":"{
('exchange_1', 'exchange_2', 'trading_pair'): [
{'p_time': 'time',
'prediction': 'result'}
]} ```

### Internal Access via AWS
The raw data and models can also be internally accessed through the Organization's AWS accounts.
-AWS RDS: RDS databases holds historical candlestick data from the cryptocurrency market APIs as well as the prediction from the models.
-AWS S3: S3 buckets holds the pickled, trained models for both Trading and Arbitrage.


## Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Please note we have a [code of conduct](./code_of_conduct.md.md). Please follow it in all your interactions with the project.

### Issue/Bug Request

 **If you are having an issue with the existing project code, please submit a bug report under the following guidelines:**
 - Check first to see if your issue has already been reported.
 - Check to see if the issue has recently been fixed by attempting to reproduce the issue using the latest master branch in the repository.
 - Create a live example of the problem.
 - Submit a detailed bug report including your environment & browser, steps to reproduce the issue, actual and expected outcomes,  where you believe the issue is originating from, and any potential solutions you have considered.

### Feature Requests

We would love to hear from you about new features which would improve this app and further the aims of our project. Please provide as much detail and information as possible to show us why you think your new feature should be implemented.

### Pull Requests

If you have developed a patch, bug fix, or new feature that would improve this app, please submit a pull request. It is best to communicate your ideas with the developers first before investing a great deal of time into a pull request to ensure that it will mesh smoothly with the project.

Remember that this project is licensed under the MIT license, and by submitting a pull request, you agree that your work will be, too.

#### Pull Request Guidelines

- Ensure any install or build dependencies are removed before the end of the layer when doing a build.
- Update the README.md with details of changes to the interface, including new plist variables, exposed ports, useful file locations and container parameters.
- Ensure that your code conforms to our existing code conventions and test coverage.
- Include the relevant issue number, if applicable.
- You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

### Attribution

These contribution guidelines have been adapted from [this good-Contributing.md-template](https://gist.github.com/PurpleBooth/b24679402957c63ec426).

## Documentation

See [Backend Documentation](_link to your backend readme here_) for details on the backend of our project.

See [Front End Documentation](_link to your front end readme here_) for details on the front end of our project.

