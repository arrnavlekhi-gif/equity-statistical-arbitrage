import itertools

import numpy as np
import pandas as pd
import yfinance as yf

from models.walkforward import build_walkforward
from strategy.execution import run_execution
from analytics.performance import stats


# ============================================================
# UNIVERSE
# ============================================================
# Embedded S&P 500 research universe.
# Snapshot date: 2026-09-21
# 503 listed securities are stored below, sorted by:
#     GICS Sector -> GICS Sub-Industry -> ticker
#
# Maintenance:
# - When S&P membership changes, edit the affected entry below.
# - Update SP500_UNIVERSE_AS_OF.
# - Keep SP500_EXPECTED_SECURITIES consistent with the snapshot.
#
# Historical backtests using this fixed snapshot retain survivorship bias.

SP500_UNIVERSE_AS_OF = "2026-09-21"
SP500_EXPECTED_SECURITIES = 503

SP500_EQUITIES_BY_SECTOR = {'Communication Services': {'Advertising': [{'ticker': 'APP', 'company': 'AppLovin', 'date_added': '2025-09-22'}, {'ticker': 'OMC', 'company': 'Omnicom Group', 'date_added': '1997-12-31'}], 'Broadcasting': [{'ticker': 'FOX', 'company': 'Fox Corporation (Class B)', 'date_added': '2019-03-19'}, {'ticker': 'FOXA', 'company': 'Fox Corporation (Class A)', 'date_added': '2019-03-19'}, {'ticker': 'WBD', 'company': 'Warner Bros. Discovery', 'date_added': '2022-04-11'}], 'Cable & Satellite': [{'ticker': 'CHTR', 'company': 'Charter Communications', 'date_added': '2016-09-08'}, {'ticker': 'CMCSA', 'company': 'Comcast', 'date_added': '2002-11-19'}], 'Integrated Telecommunication Services': [{'ticker': 'T', 'company': 'AT&T', 'date_added': '1983-11-30'}, {'ticker': 'VZ', 'company': 'Verizon', 'date_added': '1983-11-30'}], 'Interactive Home Entertainment': [{'ticker': 'TTWO', 'company': 'Take-Two Interactive', 'date_added': '2018-03-19'}], 'Interactive Media & Services': [{'ticker': 'GOOG', 'company': 'Alphabet Inc. (Class C)', 'date_added': '2014-04-03'}, {'ticker': 'GOOGL', 'company': 'Alphabet Inc. (Class A)', 'date_added': '2006-04-03'}, {'ticker': 'META', 'company': 'Meta Platforms', 'date_added': '2013-12-23'}, {'ticker': 'RDDT', 'company': 'Reddit', 'date_added': '2026-08-18'}], 'Movies & Entertainment': [{'ticker': 'DIS', 'company': 'Walt Disney Company (The)', 'date_added': '1976-06-30'}, {'ticker': 'LYV', 'company': 'Live Nation Entertainment', 'date_added': '2019-12-23'}, {'ticker': 'NFLX', 'company': 'Netflix', 'date_added': '2010-12-20'}, {'ticker': 'PSKY', 'company': 'Paramount Skydance Corporation', 'date_added': '1994-09-30'}, {'ticker': 'TKO', 'company': 'TKO Group Holdings', 'date_added': '2025-03-24'}], 'Publishing': [{'ticker': 'NWS', 'company': 'News Corp (Class B)', 'date_added': '2015-09-18'}, {'ticker': 'NWSA', 'company': 'News Corp (Class A)', 'date_added': '2013-08-01'}], 'Wireless Telecommunication Services': [{'ticker': 'ECHO', 'company': 'EchoStar', 'date_added': '2026-03-23'}, {'ticker': 'TMUS', 'company': 'T-Mobile US', 'date_added': '2019-07-15'}]}, 'Consumer Discretionary': {'Apparel Retail': [{'ticker': 'ROST', 'company': 'Ross Stores', 'date_added': '2009-12-21'}, {'ticker': 'TJX', 'company': 'TJX Companies', 'date_added': '1985-09-30'}], 'Apparel, Accessories & Luxury Goods': [{'ticker': 'LULU', 'company': 'Lululemon Athletica', 'date_added': '2023-10-18'}, {'ticker': 'NKE', 'company': 'Nike, Inc.', 'date_added': '1988-11-30'}, {'ticker': 'RL', 'company': 'Ralph Lauren Corporation', 'date_added': '2007-02-02'}, {'ticker': 'TPR', 'company': 'Tapestry, Inc.', 'date_added': '2004-09-01'}], 'Automobile Manufacturers': [{'ticker': 'F', 'company': 'Ford Motor Company', 'date_added': '1957-03-04'}, {'ticker': 'GM', 'company': 'General Motors', 'date_added': '2013-06-06'}, {'ticker': 'TSLA', 'company': 'Tesla, Inc.', 'date_added': '2020-12-21'}], 'Automotive Parts & Equipment': [{'ticker': 'APTV', 'company': 'Aptiv', 'date_added': '2012-12-24'}], 'Automotive Retail': [{'ticker': 'AZO', 'company': 'AutoZone', 'date_added': '1997-01-02'}, {'ticker': 'CVNA', 'company': 'Carvana', 'date_added': '2025-12-22'}, {'ticker': 'ORLY', 'company': "O'Reilly Automotive", 'date_added': '2009-03-27'}], 'Broadline Retail': [{'ticker': 'AMZN', 'company': 'Amazon', 'date_added': '2005-11-18'}, {'ticker': 'EBAY', 'company': 'eBay Inc.', 'date_added': '2002-07-22'}], 'Casinos & Gaming': [{'ticker': 'LVS', 'company': 'Las Vegas Sands', 'date_added': '2019-10-03'}, {'ticker': 'MGM', 'company': 'MGM Resorts', 'date_added': '2017-07-26'}, {'ticker': 'WYNN', 'company': 'Wynn Resorts', 'date_added': '2008-11-14'}], 'Computer & Electronics Retail': [{'ticker': 'BBY', 'company': 'Best Buy', 'date_added': '1999-06-29'}], 'Consumer Electronics': [{'ticker': 'GRMN', 'company': 'Garmin', 'date_added': '2012-12-12'}], 'Distributors': [{'ticker': 'GPC', 'company': 'Genuine Parts Company', 'date_added': '1973-12-31'}], 'Footwear': [{'ticker': 'DECK', 'company': 'Deckers Brands', 'date_added': '2024-03-18'}], 'Home Improvement Retail': [{'ticker': 'HD', 'company': 'Home Depot (The)', 'date_added': '1988-03-31'}, {'ticker': 'LOW', 'company': "Lowe's", 'date_added': '1984-02-29'}], 'Homebuilding': [{'ticker': 'DHI', 'company': 'D. R. Horton', 'date_added': '2005-06-22'}, {'ticker': 'LEN', 'company': 'Lennar', 'date_added': '2005-10-04'}, {'ticker': 'NVR', 'company': 'NVR, Inc.', 'date_added': '2019-09-26'}, {'ticker': 'PHM', 'company': 'PulteGroup', 'date_added': '1984-04-30'}], 'Homefurnishing Retail': [{'ticker': 'WSM', 'company': 'Williams-Sonoma, Inc.', 'date_added': '2025-03-24'}], 'Hotels, Resorts & Cruise Lines': [{'ticker': 'ABNB', 'company': 'Airbnb', 'date_added': '2023-09-18'}, {'ticker': 'BKNG', 'company': 'Booking Holdings', 'date_added': '2009-11-06'}, {'ticker': 'CCL', 'company': 'Carnival Corporation', 'date_added': '1998-12-22'}, {'ticker': 'EXPE', 'company': 'Expedia Group', 'date_added': '2007-10-02'}, {'ticker': 'HLT', 'company': 'Hilton Worldwide', 'date_added': '2017-06-19'}, {'ticker': 'MAR', 'company': 'Marriott International', 'date_added': '1998-05-29'}, {'ticker': 'NCLH', 'company': 'Norwegian Cruise Line Holdings', 'date_added': '2017-10-13'}, {'ticker': 'RCL', 'company': 'Royal Caribbean Group', 'date_added': '2014-12-05'}], 'Leisure Products': [{'ticker': 'HAS', 'company': 'Hasbro', 'date_added': '1984-09-30'}], 'Other Specialty Retail': [{'ticker': 'TSCO', 'company': 'Tractor Supply', 'date_added': '2014-01-24'}, {'ticker': 'ULTA', 'company': 'Ulta Beauty', 'date_added': '2016-04-18'}], 'Restaurants': [{'ticker': 'CMG', 'company': 'Chipotle Mexican Grill', 'date_added': '2011-04-28'}, {'ticker': 'DPZ', 'company': "Domino's", 'date_added': '2020-05-12'}, {'ticker': 'DRI', 'company': 'Darden Restaurants', 'date_added': '1995-05-31'}, {'ticker': 'MCD', 'company': "McDonald's", 'date_added': '1970-06-30'}, {'ticker': 'SBUX', 'company': 'Starbucks', 'date_added': '2000-06-07'}, {'ticker': 'YUM', 'company': 'Yum! Brands', 'date_added': '1997-10-06'}], 'Specialized Consumer Services': [{'ticker': 'DASH', 'company': 'DoorDash', 'date_added': '2025-03-24'}]}, 'Consumer Staples': {'Agricultural Products & Services': [{'ticker': 'ADM', 'company': 'Archer Daniels Midland', 'date_added': '1957-03-04'}, {'ticker': 'BG', 'company': 'Bunge Global', 'date_added': '2023-03-15'}], 'Consumer Staples Merchandise Retail': [{'ticker': 'COST', 'company': 'Costco', 'date_added': '1993-10-01'}, {'ticker': 'DG', 'company': 'Dollar General', 'date_added': '2012-12-03'}, {'ticker': 'DLTR', 'company': 'Dollar Tree', 'date_added': '2011-12-19'}, {'ticker': 'TGT', 'company': 'Target Corporation', 'date_added': '1976-12-31'}, {'ticker': 'WMT', 'company': 'Walmart', 'date_added': '1982-08-31'}], 'Distillers & Vintners': [{'ticker': 'BF.B', 'company': 'Brown–Forman', 'date_added': '1982-10-31'}, {'ticker': 'STZ', 'company': 'Constellation Brands', 'date_added': '2005-07-01'}], 'Food Distributors': [{'ticker': 'SYY', 'company': 'Sysco', 'date_added': '1986-12-31'}], 'Food Retail': [{'ticker': 'CASY', 'company': "Casey's", 'date_added': '2026-04-09'}, {'ticker': 'KR', 'company': 'Kroger', 'date_added': '1957-03-04'}], 'Household Products': [{'ticker': 'CHD', 'company': 'Church & Dwight', 'date_added': '2015-12-29'}, {'ticker': 'CL', 'company': 'Colgate-Palmolive', 'date_added': '1957-03-04'}, {'ticker': 'CLX', 'company': 'Clorox', 'date_added': '1969-03-31'}, {'ticker': 'KMB', 'company': 'Kimberly-Clark', 'date_added': '1957-03-04'}], 'Packaged Foods & Meats': [{'ticker': 'GIS', 'company': 'General Mills', 'date_added': '1957-03-04'}, {'ticker': 'HRL', 'company': 'Hormel Foods', 'date_added': '2009-03-04'}, {'ticker': 'HSY', 'company': 'Hershey Company (The)', 'date_added': '1957-03-04'}, {'ticker': 'KHC', 'company': 'Kraft Heinz', 'date_added': '2015-07-06'}, {'ticker': 'MDLZ', 'company': 'Mondelez International', 'date_added': '2012-10-02'}, {'ticker': 'MKC', 'company': 'McCormick & Company', 'date_added': '2003-03-20'}, {'ticker': 'SJM', 'company': 'J.M. Smucker Company (The)', 'date_added': '2008-11-06'}, {'ticker': 'TSN', 'company': 'Tyson Foods', 'date_added': '2005-08-10'}], 'Personal Care Products': [{'ticker': 'EL', 'company': 'Estée Lauder Companies (The)', 'date_added': '2006-01-05'}, {'ticker': 'KVUE', 'company': 'Kenvue', 'date_added': '2023-08-25'}, {'ticker': 'PG', 'company': 'Procter & Gamble', 'date_added': '1957-03-04'}], 'Soft Drinks & Non-alcoholic Beverages': [{'ticker': 'KDP', 'company': 'Keurig Dr Pepper', 'date_added': '2022-06-21'}, {'ticker': 'KO', 'company': 'Coca-Cola Company (The)', 'date_added': '1957-03-04'}, {'ticker': 'MNST', 'company': 'Monster Beverage', 'date_added': '2012-06-28'}, {'ticker': 'PEP', 'company': 'PepsiCo', 'date_added': '1957-03-04'}], 'Tobacco': [{'ticker': 'MO', 'company': 'Altria', 'date_added': '1957-03-04'}, {'ticker': 'PM', 'company': 'Philip Morris International', 'date_added': '2008-03-31'}]}, 'Energy': {'Integrated Oil & Gas': [{'ticker': 'CVX', 'company': 'Chevron Corporation', 'date_added': '1957-03-04'}, {'ticker': 'XOM', 'company': 'ExxonMobil', 'date_added': '1957-03-04'}], 'Oil & Gas Equipment & Services': [{'ticker': 'BKR', 'company': 'Baker Hughes', 'date_added': '2017-07-07'}, {'ticker': 'HAL', 'company': 'Halliburton', 'date_added': '1957-03-04'}, {'ticker': 'SLB', 'company': 'Schlumberger', 'date_added': '1957-03-04'}], 'Oil & Gas Exploration & Production': [{'ticker': 'APA', 'company': 'APA Corporation', 'date_added': '1997-07-28'}, {'ticker': 'COP', 'company': 'ConocoPhillips', 'date_added': '1957-03-04'}, {'ticker': 'DVN', 'company': 'Devon Energy', 'date_added': '2000-08-30'}, {'ticker': 'EOG', 'company': 'EOG Resources', 'date_added': '2000-11-02'}, {'ticker': 'EQT', 'company': 'EQT Corporation', 'date_added': '2022-10-03'}, {'ticker': 'EXE', 'company': 'Expand Energy', 'date_added': '2025-03-24'}, {'ticker': 'FANG', 'company': 'Diamondback Energy', 'date_added': '2018-12-03'}, {'ticker': 'OXY', 'company': 'Occidental Petroleum', 'date_added': '1957-03-04'}, {'ticker': 'TPL', 'company': 'Texas Pacific Land Corporation', 'date_added': '2024-11-26'}], 'Oil & Gas Refining & Marketing': [{'ticker': 'MPC', 'company': 'Marathon Petroleum', 'date_added': '2011-07-01'}, {'ticker': 'PSX', 'company': 'Phillips 66', 'date_added': '2012-05-01'}, {'ticker': 'VLO', 'company': 'Valero Energy', 'date_added': '2002-12-20'}], 'Oil & Gas Storage & Transportation': [{'ticker': 'KMI', 'company': 'Kinder Morgan', 'date_added': '2012-05-25'}, {'ticker': 'OKE', 'company': 'Oneok', 'date_added': '2010-03-15'}, {'ticker': 'TRGP', 'company': 'Targa Resources', 'date_added': '2022-10-12'}, {'ticker': 'WMB', 'company': 'Williams Companies', 'date_added': '1975-03-31'}]}, 'Financials': {'Asset Management & Custody Banks': [{'ticker': 'AMP', 'company': 'Ameriprise Financial', 'date_added': '2005-10-03'}, {'ticker': 'APO', 'company': 'Apollo Global Management', 'date_added': '2024-12-23'}, {'ticker': 'ARES', 'company': 'Ares Management', 'date_added': '2025-12-11'}, {'ticker': 'BEN', 'company': 'Franklin Resources', 'date_added': '1998-04-30'}, {'ticker': 'BLK', 'company': 'BlackRock', 'date_added': '2011-04-04'}, {'ticker': 'BNY', 'company': 'BNY Mellon', 'date_added': '1995-03-31'}, {'ticker': 'BX', 'company': 'Blackstone Inc.', 'date_added': '2023-09-18'}, {'ticker': 'IVZ', 'company': 'Invesco', 'date_added': '2008-08-21'}, {'ticker': 'KKR', 'company': 'KKR & Co.', 'date_added': '2024-06-24'}, {'ticker': 'NTRS', 'company': 'Northern Trust', 'date_added': '1998-01-30'}, {'ticker': 'STT', 'company': 'State Street Corporation', 'date_added': '2003-03-14'}, {'ticker': 'TROW', 'company': 'T. Rowe Price', 'date_added': '2019-07-29'}], 'Consumer Finance': [{'ticker': 'AXP', 'company': 'American Express', 'date_added': '1976-06-30'}, {'ticker': 'COF', 'company': 'Capital One', 'date_added': '1998-07-01'}, {'ticker': 'SYF', 'company': 'Synchrony Financial', 'date_added': '2015-11-18'}], 'Diversified Banks': [{'ticker': 'BAC', 'company': 'Bank of America', 'date_added': '1976-06-30'}, {'ticker': 'C', 'company': 'Citigroup', 'date_added': '1988-05-31'}, {'ticker': 'JPM', 'company': 'JPMorgan Chase', 'date_added': '1975-06-30'}, {'ticker': 'PNC', 'company': 'PNC Financial Services', 'date_added': '1988-04-30'}, {'ticker': 'TFC', 'company': 'Truist Financial', 'date_added': '1997-12-04'}, {'ticker': 'USB', 'company': 'U.S. Bancorp', 'date_added': '1999-11-01'}, {'ticker': 'WFC', 'company': 'Wells Fargo', 'date_added': '1976-06-30'}], 'Financial Exchanges & Data': [{'ticker': 'CBOE', 'company': 'Cboe Global Markets', 'date_added': '2017-03-01'}, {'ticker': 'CME', 'company': 'CME Group', 'date_added': '2006-08-11'}, {'ticker': 'COIN', 'company': 'Coinbase', 'date_added': '2025-05-19'}, {'ticker': 'FDS', 'company': 'FactSet', 'date_added': '2021-12-20'}, {'ticker': 'ICE', 'company': 'Intercontinental Exchange', 'date_added': '2007-09-26'}, {'ticker': 'MCO', 'company': "Moody's Corporation", 'date_added': '1998-07-01'}, {'ticker': 'MSCI', 'company': 'MSCI', 'date_added': '2018-04-04'}, {'ticker': 'NDAQ', 'company': 'Nasdaq, Inc.', 'date_added': '2008-10-22'}, {'ticker': 'SPGI', 'company': 'S&P Global', 'date_added': '1957-03-04'}], 'Insurance Brokers': [{'ticker': 'AJG', 'company': 'Arthur J. Gallagher & Co.', 'date_added': '2016-05-31'}, {'ticker': 'AON', 'company': 'Aon plc', 'date_added': '1996-04-23'}, {'ticker': 'BRO', 'company': 'Brown & Brown', 'date_added': '2021-09-20'}, {'ticker': 'ERIE', 'company': 'Erie Indemnity', 'date_added': '2024-09-23'}, {'ticker': 'MRSH', 'company': 'Marsh McLennan', 'date_added': '1987-08-31'}, {'ticker': 'WTW', 'company': 'Willis Towers Watson', 'date_added': '2016-01-05'}], 'Investment Banking & Brokerage': [{'ticker': 'GS', 'company': 'Goldman Sachs', 'date_added': '2002-07-22'}, {'ticker': 'HOOD', 'company': 'Robinhood Markets', 'date_added': '2025-09-22'}, {'ticker': 'IBKR', 'company': 'Interactive Brokers', 'date_added': '2025-08-28'}, {'ticker': 'MS', 'company': 'Morgan Stanley', 'date_added': '1993-07-29'}, {'ticker': 'RJF', 'company': 'Raymond James Financial', 'date_added': '2017-03-20'}, {'ticker': 'SCHW', 'company': 'Charles Schwab Corporation', 'date_added': '1997-06-02'}], 'Life & Health Insurance': [{'ticker': 'AFL', 'company': 'Aflac', 'date_added': '1999-05-28'}, {'ticker': 'GL', 'company': 'Globe Life', 'date_added': '1989-04-30'}, {'ticker': 'MET', 'company': 'MetLife', 'date_added': '2000-12-11'}, {'ticker': 'PFG', 'company': 'Principal Financial Group', 'date_added': '2002-07-22'}, {'ticker': 'PRU', 'company': 'Prudential Financial', 'date_added': '2002-07-22'}], 'Multi-Sector Holdings': [{'ticker': 'BRK.B', 'company': 'Berkshire Hathaway', 'date_added': '2010-02-16'}], 'Multi-line Insurance': [{'ticker': 'AIG', 'company': 'American International Group', 'date_added': '1980-03-31'}, {'ticker': 'AIZ', 'company': 'Assurant', 'date_added': '2007-04-10'}, {'ticker': 'L', 'company': 'Loews Corporation', 'date_added': '1995-05-31'}], 'Property & Casualty Insurance': [{'ticker': 'ACGL', 'company': 'Arch Capital Group', 'date_added': '2022-11-01'}, {'ticker': 'ALL', 'company': 'Allstate', 'date_added': '1995-07-13'}, {'ticker': 'CB', 'company': 'Chubb Limited', 'date_added': '2010-07-15'}, {'ticker': 'CINF', 'company': 'Cincinnati Financial', 'date_added': '1997-12-18'}, {'ticker': 'HIG', 'company': 'Hartford (The)', 'date_added': '1957-03-04'}, {'ticker': 'PGR', 'company': 'Progressive Corporation', 'date_added': '1997-08-04'}, {'ticker': 'TRV', 'company': 'Travelers Companies (The)', 'date_added': '2002-08-21'}, {'ticker': 'WRB', 'company': 'W. R. Berkley Corporation', 'date_added': '2019-12-05'}], 'Regional Banks': [{'ticker': 'CFG', 'company': 'Citizens Financial Group', 'date_added': '2016-01-29'}, {'ticker': 'FITB', 'company': 'Fifth Third Bancorp', 'date_added': '1996-03-29'}, {'ticker': 'HBAN', 'company': 'Huntington Bancshares', 'date_added': '1997-08-28'}, {'ticker': 'KEY', 'company': 'KeyCorp', 'date_added': '1994-03-01'}, {'ticker': 'MTB', 'company': 'M&T Bank', 'date_added': '2004-02-23'}, {'ticker': 'RF', 'company': 'Regions Financial Corporation', 'date_added': '1998-08-28'}], 'Reinsurance': [{'ticker': 'EG', 'company': 'Everest Group', 'date_added': '2017-06-19'}], 'Transaction & Payment Processing Services': [{'ticker': 'CPAY', 'company': 'Corpay', 'date_added': '2018-06-20'}, {'ticker': 'FIS', 'company': 'Fidelity National Information Services', 'date_added': '2006-11-10'}, {'ticker': 'FISV', 'company': 'Fiserv', 'date_added': '2001-04-02'}, {'ticker': 'GPN', 'company': 'Global Payments', 'date_added': '2016-04-25'}, {'ticker': 'JKHY', 'company': 'Jack Henry & Associates', 'date_added': '2018-11-13'}, {'ticker': 'MA', 'company': 'Mastercard', 'date_added': '2008-07-18'}, {'ticker': 'PYPL', 'company': 'PayPal', 'date_added': '2015-07-20'}, {'ticker': 'V', 'company': 'Visa Inc.', 'date_added': '2009-12-21'}, {'ticker': 'XYZ', 'company': 'Block, Inc.', 'date_added': '2025-07-23'}]}, 'Health Care': {'Biotechnology': [{'ticker': 'ABBV', 'company': 'AbbVie', 'date_added': '2012-12-31'}, {'ticker': 'AMGN', 'company': 'Amgen', 'date_added': '1992-01-02'}, {'ticker': 'BIIB', 'company': 'Biogen', 'date_added': '2003-11-13'}, {'ticker': 'GILD', 'company': 'Gilead Sciences', 'date_added': '2004-07-01'}, {'ticker': 'INCY', 'company': 'Incyte', 'date_added': '2017-02-28'}, {'ticker': 'MRNA', 'company': 'Moderna', 'date_added': '2021-07-21'}, {'ticker': 'REGN', 'company': 'Regeneron Pharmaceuticals', 'date_added': '2013-05-01'}, {'ticker': 'VRTX', 'company': 'Vertex Pharmaceuticals', 'date_added': '2013-09-23'}], 'Health Care Distributors': [{'ticker': 'CAH', 'company': 'Cardinal Health', 'date_added': '1997-05-27'}, {'ticker': 'COR', 'company': 'Cencora', 'date_added': '2001-08-30'}, {'ticker': 'HSIC', 'company': 'Henry Schein', 'date_added': '2015-03-17'}, {'ticker': 'MCK', 'company': 'McKesson Corporation', 'date_added': '1999-01-13'}], 'Health Care Equipment': [{'ticker': 'ABT', 'company': 'Abbott Laboratories', 'date_added': '1957-03-04'}, {'ticker': 'BAX', 'company': 'Baxter International', 'date_added': '1972-09-30'}, {'ticker': 'BDX', 'company': 'Becton Dickinson', 'date_added': '1972-09-30'}, {'ticker': 'BSX', 'company': 'Boston Scientific', 'date_added': '1995-02-24'}, {'ticker': 'DXCM', 'company': 'Dexcom', 'date_added': '2020-05-12'}, {'ticker': 'EW', 'company': 'Edwards Lifesciences', 'date_added': '2011-04-01'}, {'ticker': 'GEHC', 'company': 'GE HealthCare', 'date_added': '2023-01-04'}, {'ticker': 'IDXX', 'company': 'Idexx Laboratories', 'date_added': '2017-01-05'}, {'ticker': 'ISRG', 'company': 'Intuitive Surgical', 'date_added': '2008-06-02'}, {'ticker': 'MDT', 'company': 'Medtronic', 'date_added': '1986-10-31'}, {'ticker': 'PODD', 'company': 'Insulet Corporation', 'date_added': '2023-03-15'}, {'ticker': 'RMD', 'company': 'ResMed|', 'date_added': '2017-07-26'}, {'ticker': 'RVTY', 'company': 'Revvity', 'date_added': '1985-05-31'}, {'ticker': 'STE', 'company': 'Steris', 'date_added': '2019-12-23'}, {'ticker': 'SYK', 'company': 'Stryker Corporation', 'date_added': '2000-12-12'}, {'ticker': 'ZBH', 'company': 'Zimmer Biomet', 'date_added': '2001-08-07'}], 'Health Care Facilities': [{'ticker': 'HCA', 'company': 'HCA Healthcare', 'date_added': '2015-01-27'}, {'ticker': 'UHS', 'company': 'Universal Health Services', 'date_added': '2014-09-20'}], 'Health Care Services': [{'ticker': 'CI', 'company': 'Cigna', 'date_added': '1976-06-30'}, {'ticker': 'CVS', 'company': 'CVS Health', 'date_added': '1957-03-04'}, {'ticker': 'DGX', 'company': 'Quest Diagnostics', 'date_added': '2002-12-12'}, {'ticker': 'DVA', 'company': 'DaVita', 'date_added': '2008-07-31'}, {'ticker': 'LH', 'company': 'Labcorp', 'date_added': '2004-11-01'}], 'Health Care Supplies': [{'ticker': 'ALGN', 'company': 'Align Technology', 'date_added': '2017-06-19'}, {'ticker': 'COO', 'company': 'Cooper Companies (The)', 'date_added': '2016-09-23'}, {'ticker': 'WST', 'company': 'West Pharmaceutical Services', 'date_added': '2020-05-22'}], 'Health Care Technology': [{'ticker': 'SOLV', 'company': 'Solventum', 'date_added': '2024-04-01'}, {'ticker': 'VEEV', 'company': 'Veeva Systems', 'date_added': '2026-05-07'}], 'Life Sciences Tools & Services': [{'ticker': 'A', 'company': 'Agilent Technologies', 'date_added': '2000-06-05'}, {'ticker': 'CRL', 'company': 'Charles River Laboratories', 'date_added': '2021-05-14'}, {'ticker': 'DHR', 'company': 'Danaher Corporation', 'date_added': '1998-11-18'}, {'ticker': 'ILMN', 'company': 'Illumina, Inc.', 'date_added': '2026-09-21'}, {'ticker': 'IQV', 'company': 'IQVIA', 'date_added': '2017-08-29'}, {'ticker': 'MTD', 'company': 'Mettler Toledo', 'date_added': '2016-09-06'}, {'ticker': 'TECH', 'company': 'Bio-Techne', 'date_added': '2021-08-30'}, {'ticker': 'TMO', 'company': 'Thermo Fisher Scientific', 'date_added': '2004-08-03'}, {'ticker': 'WAT', 'company': 'Waters Corporation', 'date_added': '2002-01-02'}], 'Managed Health Care': [{'ticker': 'CNC', 'company': 'Centene Corporation', 'date_added': '2016-03-30'}, {'ticker': 'ELV', 'company': 'Elevance Health', 'date_added': '2002-07-25'}, {'ticker': 'HUM', 'company': 'Humana', 'date_added': '2012-12-10'}, {'ticker': 'UNH', 'company': 'UnitedHealth Group', 'date_added': '1994-07-01'}], 'Pharmaceuticals': [{'ticker': 'BMY', 'company': 'Bristol Myers Squibb', 'date_added': '1957-03-04'}, {'ticker': 'JNJ', 'company': 'Johnson & Johnson', 'date_added': '1973-06-30'}, {'ticker': 'LLY', 'company': 'Lilly (Eli)', 'date_added': '1970-12-31'}, {'ticker': 'MRK', 'company': 'Merck & Co.', 'date_added': '1957-03-04'}, {'ticker': 'PFE', 'company': 'Pfizer', 'date_added': '1957-03-04'}, {'ticker': 'VTRS', 'company': 'Viatris', 'date_added': '2004-04-23'}, {'ticker': 'ZTS', 'company': 'Zoetis', 'date_added': '2013-06-21'}]}, 'Industrials': {'Aerospace & Defense': [{'ticker': 'AXON', 'company': 'Axon Enterprise', 'date_added': '2023-05-04'}, {'ticker': 'BA', 'company': 'Boeing', 'date_added': '1957-03-04'}, {'ticker': 'GD', 'company': 'General Dynamics', 'date_added': '1957-03-04'}, {'ticker': 'GE', 'company': 'GE Aerospace', 'date_added': '1957-03-04'}, {'ticker': 'HII', 'company': 'Huntington Ingalls Industries', 'date_added': '2018-01-03'}, {'ticker': 'HONA', 'company': 'Honeywell Aerospace', 'date_added': '2026-06-29'}, {'ticker': 'HWM', 'company': 'Howmet Aerospace', 'date_added': '2016-10-21'}, {'ticker': 'LHX', 'company': 'L3Harris', 'date_added': '2008-09-22'}, {'ticker': 'LMT', 'company': 'Lockheed Martin', 'date_added': '1957-03-04'}, {'ticker': 'NOC', 'company': 'Northrop Grumman', 'date_added': '1957-03-04'}, {'ticker': 'RTX', 'company': 'RTX Corporation', 'date_added': '1957-03-04'}, {'ticker': 'TDG', 'company': 'TransDigm Group', 'date_added': '2016-06-03'}, {'ticker': 'TXT', 'company': 'Textron', 'date_added': '1978-12-31'}], 'Agricultural & Farm Machinery': [{'ticker': 'DE', 'company': 'Deere & Company', 'date_added': '1957-03-04'}], 'Air Freight & Logistics': [{'ticker': 'CHRW', 'company': 'C.H. Robinson', 'date_added': '2007-03-02'}, {'ticker': 'EXPD', 'company': 'Expeditors International', 'date_added': '2007-10-10'}, {'ticker': 'FDX', 'company': 'FedEx', 'date_added': '1980-12-31'}, {'ticker': 'UPS', 'company': 'United Parcel Service', 'date_added': '2002-07-22'}], 'Building Products': [{'ticker': 'ALLE', 'company': 'Allegion', 'date_added': '2013-12-02'}, {'ticker': 'AOS', 'company': 'A. O. Smith', 'date_added': '2017-07-26'}, {'ticker': 'CARR', 'company': 'Carrier Global', 'date_added': '2020-04-03'}, {'ticker': 'JCI', 'company': 'Johnson Controls', 'date_added': '2010-08-27'}, {'ticker': 'LII', 'company': 'Lennox International', 'date_added': '2024-12-23'}, {'ticker': 'MAS', 'company': 'Masco', 'date_added': '1981-06-30'}, {'ticker': 'TT', 'company': 'Trane Technologies', 'date_added': '2010-11-17'}], 'Cargo Ground Transportation': [{'ticker': 'FDXF', 'company': 'FedEx Freight', 'date_added': '2026-06-01'}, {'ticker': 'JBHT', 'company': 'J.B. Hunt', 'date_added': '2015-07-01'}, {'ticker': 'ODFL', 'company': 'Old Dominion', 'date_added': '2019-12-09'}], 'Construction & Engineering': [{'ticker': 'EME', 'company': 'Emcor', 'date_added': '2025-09-22'}, {'ticker': 'FIX', 'company': 'Comfort Systems USA', 'date_added': '2025-12-22'}, {'ticker': 'J', 'company': 'Jacobs Solutions', 'date_added': '2007-10-26'}, {'ticker': 'PWR', 'company': 'Quanta Services', 'date_added': '2009-07-01'}], 'Construction Machinery & Heavy Transportation Equipment': [{'ticker': 'CAT', 'company': 'Caterpillar Inc.', 'date_added': '1957-03-04'}, {'ticker': 'CMI', 'company': 'Cummins', 'date_added': '1965-03-31'}, {'ticker': 'PCAR', 'company': 'Paccar', 'date_added': '1980-12-31'}, {'ticker': 'WAB', 'company': 'Wabtec', 'date_added': '2019-02-27'}], 'Data Processing & Outsourced Services': [{'ticker': 'BR', 'company': 'Broadridge Financial Solutions', 'date_added': '2018-06-18'}], 'Diversified Support Services': [{'ticker': 'CPRT', 'company': 'Copart', 'date_added': '2018-07-02'}, {'ticker': 'CTAS', 'company': 'Cintas', 'date_added': '2001-03-01'}, {'ticker': 'LDOS', 'company': 'Leidos', 'date_added': '2019-08-09'}], 'Electrical Components & Equipment': [{'ticker': 'AME', 'company': 'Ametek', 'date_added': '2013-09-23'}, {'ticker': 'BE', 'company': 'Bloom Energy', 'date_added': '2026-09-21'}, {'ticker': 'EMR', 'company': 'Emerson Electric', 'date_added': '1965-03-31'}, {'ticker': 'ETN', 'company': 'Eaton Corporation', 'date_added': '1957-03-04'}, {'ticker': 'ROK', 'company': 'Rockwell Automation', 'date_added': '2000-03-12'}, {'ticker': 'VRT', 'company': 'Vertiv', 'date_added': '2026-03-23'}], 'Environmental & Facilities Services': [{'ticker': 'ROL', 'company': 'Rollins, Inc.', 'date_added': '2018-10-01'}, {'ticker': 'RSG', 'company': 'Republic Services', 'date_added': '2008-12-05'}, {'ticker': 'VLTO', 'company': 'Veralto', 'date_added': '2023-10-02'}, {'ticker': 'WM', 'company': 'Waste Management', 'date_added': '1998-08-31'}], 'Heavy Electrical Equipment': [{'ticker': 'GEV', 'company': 'GE Vernova', 'date_added': '2024-04-02'}, {'ticker': 'GNRC', 'company': 'Generac', 'date_added': '2021-03-22'}], 'Human Resource & Employment Services': [{'ticker': 'ADP', 'company': 'Automatic Data Processing', 'date_added': '1981-03-31'}, {'ticker': 'PAYX', 'company': 'Paychex', 'date_added': '1998-10-01'}], 'Industrial Conglomerates': [{'ticker': 'DD', 'company': 'DuPont', 'date_added': '2019-06-03'}, {'ticker': 'HON', 'company': 'Honeywell Technologies', 'date_added': '1957-03-04'}, {'ticker': 'MMM', 'company': '3M', 'date_added': '1957-03-04'}], 'Industrial Machinery & Supplies & Components': [{'ticker': 'DOV', 'company': 'Dover Corporation', 'date_added': '1985-10-31'}, {'ticker': 'FTV', 'company': 'Fortive', 'date_added': '2016-07-01'}, {'ticker': 'GWW', 'company': 'W. W. Grainger', 'date_added': '1981-06-30'}, {'ticker': 'HUBB', 'company': 'Hubbell Incorporated', 'date_added': '2023-10-18'}, {'ticker': 'IEX', 'company': 'IDEX Corporation', 'date_added': '2019-08-09'}, {'ticker': 'IR', 'company': 'Ingersoll Rand', 'date_added': '2020-03-03'}, {'ticker': 'ITW', 'company': 'Illinois Tool Works', 'date_added': '1986-02-28'}, {'ticker': 'NDSN', 'company': 'Nordson Corporation', 'date_added': '2022-02-15'}, {'ticker': 'OTIS', 'company': 'Otis Worldwide', 'date_added': '2020-04-03'}, {'ticker': 'PH', 'company': 'Parker Hannifin', 'date_added': '1985-11-30'}, {'ticker': 'PNR', 'company': 'Pentair', 'date_added': '2012-10-01'}, {'ticker': 'SNA', 'company': 'Snap-on', 'date_added': '1982-09-30'}, {'ticker': 'SWK', 'company': 'Stanley Black & Decker', 'date_added': '1982-09-30'}, {'ticker': 'XYL', 'company': 'Xylem Inc.', 'date_added': '2011-11-01'}], 'Passenger Airlines': [{'ticker': 'DAL', 'company': 'Delta Air Lines', 'date_added': '2013-09-11'}, {'ticker': 'LUV', 'company': 'Southwest Airlines', 'date_added': '1994-07-01'}, {'ticker': 'UAL', 'company': 'United Airlines Holdings', 'date_added': '2015-09-03'}], 'Passenger Ground Transportation': [{'ticker': 'UBER', 'company': 'Uber', 'date_added': '2023-12-18'}], 'Rail Transportation': [{'ticker': 'CSX', 'company': 'CSX Corporation', 'date_added': '1957-03-04'}, {'ticker': 'NSC', 'company': 'Norfolk Southern', 'date_added': '1957-03-04'}, {'ticker': 'UNP', 'company': 'Union Pacific Corporation', 'date_added': '1957-03-04'}], 'Research & Consulting Services': [{'ticker': 'EFX', 'company': 'Equifax', 'date_added': '1997-06-19'}, {'ticker': 'VRSK', 'company': 'Verisk Analytics', 'date_added': '2015-10-08'}], 'Trading Companies & Distributors': [{'ticker': 'FAST', 'company': 'Fastenal', 'date_added': '2008-09-15'}, {'ticker': 'FERG', 'company': 'Ferguson Enterprises', 'date_added': '2026-08-05'}, {'ticker': 'URI', 'company': 'United Rentals', 'date_added': '2014-09-20'}]}, 'Information Technology': {'Application Software': [{'ticker': 'ADBE', 'company': 'Adobe Inc.', 'date_added': '1997-05-05'}, {'ticker': 'ADSK', 'company': 'Autodesk', 'date_added': '1989-12-01'}, {'ticker': 'CDNS', 'company': 'Cadence Design Systems', 'date_added': '2017-09-18'}, {'ticker': 'CRM', 'company': 'Salesforce', 'date_added': '2008-09-15'}, {'ticker': 'DDOG', 'company': 'Datadog', 'date_added': '2025-07-09'}, {'ticker': 'FICO', 'company': 'Fair Isaac', 'date_added': '2023-03-20'}, {'ticker': 'INTU', 'company': 'Intuit', 'date_added': '2000-12-05'}, {'ticker': 'ORCL', 'company': 'Oracle Corporation', 'date_added': '1989-08-31'}, {'ticker': 'PLTR', 'company': 'Palantir Technologies', 'date_added': '2024-09-23'}, {'ticker': 'PTC', 'company': 'PTC Inc.', 'date_added': '2021-04-20'}, {'ticker': 'SNPS', 'company': 'Synopsys', 'date_added': '2017-03-16'}, {'ticker': 'TRMB', 'company': 'Trimble Inc.', 'date_added': '2021-01-21'}, {'ticker': 'TYL', 'company': 'Tyler Technologies', 'date_added': '2020-06-22'}, {'ticker': 'WDAY', 'company': 'Workday, Inc.', 'date_added': '2024-12-23'}], 'Communications Equipment': [{'ticker': 'ANET', 'company': 'Arista Networks', 'date_added': '2018-08-28'}, {'ticker': 'CIEN', 'company': 'Ciena', 'date_added': '2026-02-09'}, {'ticker': 'CSCO', 'company': 'Cisco', 'date_added': '1993-12-01'}, {'ticker': 'FFIV', 'company': 'F5, Inc.', 'date_added': '2010-12-20'}, {'ticker': 'LITE', 'company': 'Lumentum', 'date_added': '2026-03-23'}, {'ticker': 'MSI', 'company': 'Motorola Solutions', 'date_added': '1957-03-04'}], 'Electronic Components': [{'ticker': 'APH', 'company': 'Amphenol', 'date_added': '2008-09-30'}, {'ticker': 'COHR', 'company': 'Coherent Corp.', 'date_added': '2026-03-23'}, {'ticker': 'GLW', 'company': 'Corning Inc.', 'date_added': '1995-02-27'}], 'Electronic Equipment & Instruments': [{'ticker': 'KEYS', 'company': 'Keysight Technologies', 'date_added': '2018-11-06'}, {'ticker': 'ROP', 'company': 'Roper Technologies', 'date_added': '2009-12-23'}, {'ticker': 'TDY', 'company': 'Teledyne Technologies', 'date_added': '2020-06-22'}, {'ticker': 'ZBRA', 'company': 'Zebra Technologies', 'date_added': '2019-12-23'}], 'Electronic Manufacturing Services': [{'ticker': 'FLEX', 'company': 'Flex Ltd.', 'date_added': '2026-06-22'}, {'ticker': 'JBL', 'company': 'Jabil', 'date_added': '2023-12-18'}, {'ticker': 'TEL', 'company': 'TE Connectivity', 'date_added': '2011-10-17'}], 'IT Consulting & Other Services': [{'ticker': 'ACN', 'company': 'Accenture', 'date_added': '2011-07-06'}, {'ticker': 'CTSH', 'company': 'Cognizant', 'date_added': '2006-11-17'}, {'ticker': 'IBM', 'company': 'IBM', 'date_added': '1957-03-04'}, {'ticker': 'IT', 'company': 'Gartner', 'date_added': '2017-04-05'}], 'Internet Services & Infrastructure': [{'ticker': 'AKAM', 'company': 'Akamai Technologies', 'date_added': '2007-07-12'}, {'ticker': 'GDDY', 'company': 'GoDaddy', 'date_added': '2024-06-24'}, {'ticker': 'VRSN', 'company': 'Verisign', 'date_added': '2006-02-01'}], 'Semiconductor Materials & Equipment': [{'ticker': 'AMAT', 'company': 'Applied Materials', 'date_added': '1995-03-16'}, {'ticker': 'KLAC', 'company': 'KLA Corporation', 'date_added': '1997-09-30'}, {'ticker': 'LRCX', 'company': 'Lam Research', 'date_added': '2012-06-29'}, {'ticker': 'Q', 'company': 'Qnity Electronics', 'date_added': '2025-11-03'}, {'ticker': 'TER', 'company': 'Teradyne', 'date_added': '2020-09-21'}], 'Semiconductors': [{'ticker': 'ADI', 'company': 'Analog Devices', 'date_added': '1999-10-12'}, {'ticker': 'AMD', 'company': 'Advanced Micro Devices', 'date_added': '2017-03-20'}, {'ticker': 'AVGO', 'company': 'Broadcom', 'date_added': '2014-05-08'}, {'ticker': 'FSLR', 'company': 'First Solar', 'date_added': '2022-12-19'}, {'ticker': 'INTC', 'company': 'Intel', 'date_added': '1976-12-31'}, {'ticker': 'MCHP', 'company': 'Microchip Technology', 'date_added': '2007-09-07'}, {'ticker': 'MPWR', 'company': 'Monolithic Power Systems', 'date_added': '2021-02-12'}, {'ticker': 'MRVL', 'company': 'Marvell Technology', 'date_added': '2026-06-22'}, {'ticker': 'MU', 'company': 'Micron Technology', 'date_added': '1994-09-27'}, {'ticker': 'NVDA', 'company': 'Nvidia', 'date_added': '2001-11-30'}, {'ticker': 'NXPI', 'company': 'NXP Semiconductors', 'date_added': '2021-03-22'}, {'ticker': 'ON', 'company': 'ON Semiconductor', 'date_added': '2022-06-21'}, {'ticker': 'QCOM', 'company': 'Qualcomm', 'date_added': '1999-07-22'}, {'ticker': 'SWKS', 'company': 'Skyworks Solutions', 'date_added': '2015-03-12'}, {'ticker': 'TXN', 'company': 'Texas Instruments', 'date_added': '2001-03-12'}], 'Systems Software': [{'ticker': 'CRWD', 'company': 'CrowdStrike', 'date_added': '2024-06-24'}, {'ticker': 'FTNT', 'company': 'Fortinet', 'date_added': '2018-10-11'}, {'ticker': 'GEN', 'company': 'Gen Digital', 'date_added': '2003-03-25'}, {'ticker': 'MSFT', 'company': 'Microsoft', 'date_added': '1994-06-01'}, {'ticker': 'NOW', 'company': 'ServiceNow', 'date_added': '2019-11-21'}, {'ticker': 'PANW', 'company': 'Palo Alto Networks', 'date_added': '2023-06-20'}], 'Technology Distributors': [{'ticker': 'CDW', 'company': 'CDW Corporation', 'date_added': '2019-09-23'}], 'Technology Hardware, Storage & Peripherals': [{'ticker': 'AAPL', 'company': 'Apple Inc.', 'date_added': '1982-11-30'}, {'ticker': 'DELL', 'company': 'Dell Technologies', 'date_added': '2024-09-23'}, {'ticker': 'HPE', 'company': 'Hewlett Packard Enterprise', 'date_added': '2015-11-02'}, {'ticker': 'HPQ', 'company': 'HP Inc.', 'date_added': '1974-12-31'}, {'ticker': 'NTAP', 'company': 'NetApp', 'date_added': '1999-06-25'}, {'ticker': 'P', 'company': 'Everpure', 'date_added': '2026-09-21'}, {'ticker': 'SMCI', 'company': 'Supermicro', 'date_added': '2024-03-18'}, {'ticker': 'SNDK', 'company': 'Sandisk', 'date_added': '2025-11-28'}, {'ticker': 'STX', 'company': 'Seagate Technology', 'date_added': '2012-07-02'}, {'ticker': 'WDC', 'company': 'Western Digital', 'date_added': '2009-07-01'}]}, 'Materials': {'Commodity Chemicals': [{'ticker': 'DOW', 'company': 'Dow Inc.', 'date_added': '2019-04-01'}], 'Construction Materials': [{'ticker': 'CRH', 'company': 'CRH plc', 'date_added': '2025-12-22'}, {'ticker': 'MLM', 'company': 'Martin Marietta Materials', 'date_added': '2014-07-02'}, {'ticker': 'VMC', 'company': 'Vulcan Materials Company', 'date_added': '1999-06-30'}], 'Copper': [{'ticker': 'FCX', 'company': 'Freeport-McMoRan', 'date_added': '2011-07-01'}], 'Fertilizers & Agricultural Chemicals': [{'ticker': 'CF', 'company': 'CF Industries', 'date_added': '2008-08-27'}, {'ticker': 'CTVA', 'company': 'Corteva', 'date_added': '2019-06-03'}, {'ticker': 'MOS', 'company': 'Mosaic Company (The)', 'date_added': '2011-09-26'}], 'Gold': [{'ticker': 'NEM', 'company': 'Newmont', 'date_added': '1969-06-30'}], 'Industrial Gases': [{'ticker': 'APD', 'company': 'Air Products', 'date_added': '1985-04-30'}, {'ticker': 'LIN', 'company': 'Linde plc', 'date_added': '1992-07-01'}], 'Metal, Glass & Plastic Containers': [{'ticker': 'BALL', 'company': 'Ball Corporation', 'date_added': '1984-10-31'}], 'Paper & Plastic Packaging Products & Materials': [{'ticker': 'AMCR', 'company': 'Amcor', 'date_added': '2019-06-07'}, {'ticker': 'AVY', 'company': 'Avery Dennison', 'date_added': '1987-12-31'}, {'ticker': 'IP', 'company': 'International Paper', 'date_added': '1957-03-04'}, {'ticker': 'PKG', 'company': 'Packaging Corporation of America', 'date_added': '2017-07-26'}, {'ticker': 'SW', 'company': 'Smurfit Westrock', 'date_added': '2024-07-08'}], 'Specialty Chemicals': [{'ticker': 'ALB', 'company': 'Albemarle Corporation', 'date_added': '2016-07-01'}, {'ticker': 'ECL', 'company': 'Ecolab', 'date_added': '1989-01-31'}, {'ticker': 'IFF', 'company': 'International Flavors & Fragrances', 'date_added': '1976-03-31'}, {'ticker': 'LYB', 'company': 'LyondellBasell', 'date_added': '2012-09-05'}, {'ticker': 'PPG', 'company': 'PPG Industries', 'date_added': '1957-03-04'}, {'ticker': 'SHW', 'company': 'Sherwin-Williams', 'date_added': '1964-06-30'}], 'Steel': [{'ticker': 'NUE', 'company': 'Nucor', 'date_added': '1985-04-30'}, {'ticker': 'STLD', 'company': 'Steel Dynamics', 'date_added': '2022-12-22'}]}, 'Real Estate': {'Data Center REITs': [{'ticker': 'DLR', 'company': 'Digital Realty', 'date_added': '2016-05-18'}, {'ticker': 'EQIX', 'company': 'Equinix', 'date_added': '2015-03-20'}], 'Health Care REITs': [{'ticker': 'DOC', 'company': 'Healthpeak Properties', 'date_added': '2008-03-31'}, {'ticker': 'VTR', 'company': 'Ventas', 'date_added': '2009-03-04'}, {'ticker': 'WELL', 'company': 'Welltower', 'date_added': '2009-01-30'}], 'Hotel & Resort REITs': [{'ticker': 'HST', 'company': 'Host Hotels & Resorts', 'date_added': '2007-03-20'}, {'ticker': 'VICI', 'company': 'Vici Properties', 'date_added': '2022-06-08'}], 'Industrial REITs': [{'ticker': 'PLD', 'company': 'Prologis', 'date_added': '2003-07-17'}], 'Multi-Family Residential REITs': [{'ticker': 'CPT', 'company': 'Camden Property Trust', 'date_added': '2022-04-04'}, {'ticker': 'ESS', 'company': 'Essex Property Trust', 'date_added': '2014-04-02'}, {'ticker': 'MAA', 'company': 'Mid-America Apartment Communities', 'date_added': '2016-12-02'}, {'ticker': 'UDR', 'company': 'UDR, Inc.', 'date_added': '2016-03-07'}, {'ticker': 'VMRK', 'company': 'Vivmark Residential', 'date_added': '2001-12-03'}], 'Office REITs': [{'ticker': 'ARE', 'company': 'Alexandria Real Estate Equities', 'date_added': '2017-03-20'}, {'ticker': 'BXP', 'company': 'BXP, Inc.', 'date_added': '2006-04-03'}], 'Other Specialized REITs': [{'ticker': 'IRM', 'company': 'Iron Mountain', 'date_added': '2009-01-06'}], 'Real Estate Services': [{'ticker': 'CBRE', 'company': 'CBRE Group', 'date_added': '2006-11-10'}, {'ticker': 'CSGP', 'company': 'CoStar Group', 'date_added': '2022-09-19'}], 'Retail REITs': [{'ticker': 'FRT', 'company': 'Federal Realty Investment Trust', 'date_added': '2016-02-01'}, {'ticker': 'KIM', 'company': 'Kimco Realty', 'date_added': '2006-04-04'}, {'ticker': 'O', 'company': 'Realty Income', 'date_added': '2015-04-07'}, {'ticker': 'REG', 'company': 'Regency Centers', 'date_added': '2017-03-02'}, {'ticker': 'SPG', 'company': 'Simon Property Group', 'date_added': '2002-06-26'}], 'Self-Storage REITs': [{'ticker': 'EXR', 'company': 'Extra Space Storage', 'date_added': '2016-01-19'}, {'ticker': 'PSA', 'company': 'Public Storage', 'date_added': '2005-08-19'}], 'Single-Family Residential REITs': [{'ticker': 'INVH', 'company': 'Invitation Homes', 'date_added': '2022-09-19'}], 'Telecom Tower REITs': [{'ticker': 'AMT', 'company': 'American Tower', 'date_added': '2007-11-19'}, {'ticker': 'CCI', 'company': 'Crown Castle', 'date_added': '2012-03-14'}, {'ticker': 'SBAC', 'company': 'SBA Communications', 'date_added': '2017-09-01'}], 'Timber REITs': [{'ticker': 'WY', 'company': 'Weyerhaeuser', 'date_added': '1979-10-01'}]}, 'Utilities': {'Electric Utilities': [{'ticker': 'AEP', 'company': 'American Electric Power', 'date_added': '1957-03-04'}, {'ticker': 'CEG', 'company': 'Constellation Energy', 'date_added': '2022-02-02'}, {'ticker': 'DUK', 'company': 'Duke Energy', 'date_added': '1976-06-30'}, {'ticker': 'EIX', 'company': 'Edison International', 'date_added': '1957-03-04'}, {'ticker': 'ES', 'company': 'Eversource Energy', 'date_added': '2009-07-24'}, {'ticker': 'ETR', 'company': 'Entergy', 'date_added': '1957-03-04'}, {'ticker': 'EVRG', 'company': 'Evergy', 'date_added': '2018-06-05'}, {'ticker': 'EXC', 'company': 'Exelon', 'date_added': '1957-03-04'}, {'ticker': 'FE', 'company': 'FirstEnergy', 'date_added': '1997-11-28'}, {'ticker': 'LNT', 'company': 'Alliant Energy', 'date_added': '2016-07-01'}, {'ticker': 'PEG', 'company': 'Public Service Enterprise Group', 'date_added': '1957-03-04'}, {'ticker': 'PPL', 'company': 'PPL Corporation', 'date_added': '2001-10-01'}, {'ticker': 'SO', 'company': 'Southern Company', 'date_added': '1957-03-04'}, {'ticker': 'VST', 'company': 'Vistra Corp.', 'date_added': '2024-05-08'}, {'ticker': 'WEC', 'company': 'WEC Energy Group', 'date_added': '2008-10-31'}], 'Gas Utilities': [{'ticker': 'ATO', 'company': 'Atmos Energy', 'date_added': '2019-02-15'}], 'Independent Power Producers & Energy Traders': [{'ticker': 'AES', 'company': 'AES Corporation', 'date_added': '1998-10-02'}, {'ticker': 'NRG', 'company': 'NRG Energy', 'date_added': '2010-01-29'}], 'Multi-Utilities': [{'ticker': 'AEE', 'company': 'Ameren', 'date_added': '1991-09-19'}, {'ticker': 'CMS', 'company': 'CMS Energy', 'date_added': '1957-03-04'}, {'ticker': 'CNP', 'company': 'CenterPoint Energy', 'date_added': '1985-07-31'}, {'ticker': 'D', 'company': 'Dominion Energy', 'date_added': '2016-11-30'}, {'ticker': 'DTE', 'company': 'DTE Energy', 'date_added': '1957-03-04'}, {'ticker': 'ED', 'company': 'Consolidated Edison', 'date_added': '1957-03-04'}, {'ticker': 'NEE', 'company': 'NextEra Energy', 'date_added': '1976-06-30'}, {'ticker': 'NI', 'company': 'NiSource', 'date_added': '2000-11-02'}, {'ticker': 'PCG', 'company': 'PG&E Corporation', 'date_added': '2022-10-03'}, {'ticker': 'PNW', 'company': 'Pinnacle West Capital', 'date_added': '1999-10-04'}, {'ticker': 'SRE', 'company': 'Sempra', 'date_added': '2017-03-17'}, {'ticker': 'XEL', 'company': 'Xcel Energy', 'date_added': '1957-03-04'}], 'Water Utilities': [{'ticker': 'AWK', 'company': 'American Water Works', 'date_added': '2016-03-04'}]}}

_UNIVERSE_CACHE = None
_UNIVERSE_METADATA = None


def _yahoo_symbol(symbol):
    return str(symbol).strip().replace(".", "-")


def _load_sp500_peer_groups():
    global _UNIVERSE_CACHE, _UNIVERSE_METADATA
    if _UNIVERSE_CACHE is not None:
        return _UNIVERSE_CACHE

    rows = []
    for sector, subindustries in SP500_EQUITIES_BY_SECTOR.items():
        for subindustry, equities in subindustries.items():
            for equity in equities:
                rows.append({
                    "Original Symbol": equity["ticker"],
                    "Symbol": _yahoo_symbol(equity["ticker"]),
                    "Security": equity["company"],
                    "GICS Sector": sector,
                    "GICS Sub-Industry": subindustry,
                    "Date added": equity["date_added"],
                })

    meta = pd.DataFrame(rows)

    raw_count = meta["Original Symbol"].nunique()
    if len(meta) != SP500_EXPECTED_SECURITIES or raw_count != SP500_EXPECTED_SECURITIES:
        raise ValueError(
            f"Embedded S&P universe validation failed: expected "
            f"{SP500_EXPECTED_SECURITIES} securities, found "
            f"{len(meta)} rows / {raw_count} unique symbols."
        )

    # Use narrow GICS sub-industries whenever they contain at least two names.
    sub_counts = meta.groupby("GICS Sub-Industry")["Symbol"].transform("nunique")
    meta["Peer Group"] = meta["GICS Sub-Industry"].astype(str)

    # Preserve singleton sub-industries by pooling them only with other
    # singleton sub-industries from the same GICS sector.
    singleton = sub_counts < 2
    meta.loc[singleton, "Peer Group"] = (
        meta.loc[singleton, "GICS Sector"].astype(str)
        + " — Singleton peers"
    )

    groups = {}
    for name, group in meta.groupby("Peer Group"):
        tickers = sorted(group["Symbol"].drop_duplicates().tolist())
        if len(tickers) >= 2:
            groups[str(name)] = tickers

    eligible = set(t for tickers in groups.values() for t in tickers)
    meta["Pair Eligible"] = meta["Symbol"].isin(eligible)

    _UNIVERSE_CACHE = dict(sorted(groups.items()))
    _UNIVERSE_METADATA = meta.sort_values(
        ["GICS Sector", "GICS Sub-Industry", "Symbol"]
    ).reset_index(drop=True)
    return _UNIVERSE_CACHE


def get_universe_source():
    return (
        f"Embedded S&P 500 snapshot {SP500_UNIVERSE_AS_OF} "
        f"({SP500_EXPECTED_SECURITIES} securities)"
    )


def get_universe_snapshot_date():
    return SP500_UNIVERSE_AS_OF


def get_universe_metadata(industry=None):
    _load_sp500_peer_groups()
    meta = _UNIVERSE_METADATA.copy()
    if industry is not None:
        meta = meta.loc[meta["Peer Group"] == industry].copy()
    return meta.reset_index(drop=True)


def get_universe_security_count():
    _load_sp500_peer_groups()
    return int(_UNIVERSE_METADATA["Symbol"].nunique())


def get_pair_eligible_security_count():
    _load_sp500_peer_groups()
    return int(
        _UNIVERSE_METADATA.loc[
            _UNIVERSE_METADATA["Pair Eligible"], "Symbol"
        ].nunique()
    )


def get_industries():
    return list(_load_sp500_peer_groups().keys())


def get_industry_tickers(industry):
    universes = _load_sp500_peer_groups()
    if industry not in universes:
        raise ValueError(f"Unknown peer group: {industry}")
    return universes[industry]


def generate_industry_pairs(industry):
    return list(itertools.combinations(get_industry_tickers(industry), 2))


def download_industry(industry, start, end):
    tickers = get_industry_tickers(industry)
    data = yf.download(
        tickers, start=start, end=end + pd.Timedelta(days=1),
        auto_adjust=True, progress=False, group_by="column", threads=True,
    )
    if data.empty:
        raise ValueError("No equity data could be downloaded.")
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Unexpected Yahoo Finance data format.")
    opens = data["Open"].copy()
    closes = data["Close"].copy()
    return opens, closes


def prepare_pair_data(opens, closes, ticker_a, ticker_b):
    for ticker, frame, field in [
        (ticker_a, opens, "Open"), (ticker_b, opens, "Open"),
        (ticker_a, closes, "Close"), (ticker_b, closes, "Close"),
    ]:
        if ticker not in frame.columns:
            raise ValueError(f"Missing {field} data for {ticker}.")
    return pd.concat([
        opens[ticker_a].rename("open_a"), opens[ticker_b].rename("open_b"),
        closes[ticker_a].rename("close_a"), closes[ticker_b].rename("close_b"),
    ], axis=1).dropna()


# ============================================================
# FAST PRE-SCREEN
# ============================================================
# This stage is deliberately cheap. It is a computational filter only;
# it is NOT evidence of cointegration and is not used as a performance
# criterion. The full walk-forward Engle-Granger model remains the
# statistical test for surviving pairs.

def prescreen_industry_pairs(
    closes, pairs, min_observations=252, min_return_correlation=0.20,
    min_log_price_correlation=0.60, max_pairs=25,
):
    rows = []
    for ticker_a, ticker_b in pairs:
        try:
            px = pd.concat([closes[ticker_a], closes[ticker_b]], axis=1).dropna()
            if len(px) < int(min_observations):
                continue
            px.columns = ["a", "b"]
            logp = np.log(px)
            rets = logp.diff().dropna()
            if len(rets) < int(min_observations) - 1:
                continue
            price_corr = float(logp["a"].corr(logp["b"]))
            return_corr = float(rets["a"].corr(rets["b"]))
            if not np.isfinite(price_corr) or not np.isfinite(return_corr):
                continue
            if abs(price_corr) < float(min_log_price_correlation):
                continue
            if return_corr < float(min_return_correlation):
                continue
            # Ranking is statistical/economic similarity only; no Sharpe/return.
            score = 0.55 * abs(price_corr) + 0.45 * max(return_corr, 0.0)
            rows.append({
                "ticker_a": ticker_a, "ticker_b": ticker_b,
                "pair": f"{ticker_a} / {ticker_b}",
                "log_price_correlation": price_corr,
                "return_correlation": return_corr,
                "prescreen_score": score,
            })
        except Exception:
            continue

    screen = pd.DataFrame(rows)
    if screen.empty:
        return [], screen
    screen = screen.sort_values(
        ["prescreen_score", "return_correlation"], ascending=[False, False]
    ).reset_index(drop=True)
    if max_pairs is not None and int(max_pairs) > 0:
        screen = screen.head(int(max_pairs)).copy()
    survivors = list(zip(screen["ticker_a"], screen["ticker_b"]))
    return survivors, screen


# ============================================================
# EXTRACT ONE OBSERVATION PER WALK-FORWARD BLOCK
# ============================================================

def extract_blocks(model):

    tested = model[
        model[
            "cointegration_pvalue"
        ].notna()
    ].copy()

    if tested.empty:

        return tested

    # The formation-period statistics are repeated for each
    # day of the next trading block.
    #
    # A new block begins whenever these estimated model
    # parameters change.

    comparison_columns = [
        "alpha",
        "beta",
        "cointegration_pvalue",
        "half_life",
        "orientation",
    ]

    block_change = (
        tested[
            comparison_columns
        ]
        .astype(str)
        .ne(
            tested[
                comparison_columns
            ]
            .astype(str)
            .shift()
        )
        .any(axis=1)
    )

    blocks = tested.loc[
        block_change
    ].copy()

    return blocks


# ============================================================
# COUNT DISTINCT VALIDITY REGIMES
# ============================================================

def count_validity_regimes(
    valid_series
):

    valid = (
        valid_series
        .fillna(False)
        .astype(bool)
    )

    if len(valid) == 0:

        return 0

    regime_starts = (
        valid
        &
        ~valid.shift(
            1,
            fill_value=False,
        )
    )

    return int(
        regime_starts.sum()
    )


# ============================================================
# PAST-ONLY COINTEGRATION PERSISTENCE FILTER
# ============================================================

def apply_persistence_filter(model, persistence_lookback=6, persistence_min_passes=3):
    """Require current validity plus enough PASS results in PRIOR blocks only.

    The current block is never counted toward its own persistence history.
    With lookback=6 and min_passes=3, a block can trade only when the current
    formation window passes the normal cointegration/half-life rule AND at least
    3 of the previous 6 formation windows also passed. The first 6 tested blocks
    are therefore warm-up blocks and cannot trade.
    """
    out = model.copy()
    tested = out["cointegration_pvalue"].notna()
    if not tested.any():
        out["raw_pair_valid"] = out["pair_valid"].fillna(False).astype(bool)
        out["persistence_passes"] = np.nan
        return out

    comparison_columns = ["alpha", "beta", "cointegration_pvalue", "half_life", "orientation"]
    sig = out.loc[tested, comparison_columns].astype(str)
    block_start = sig.ne(sig.shift()).any(axis=1)
    block_id = block_start.cumsum()

    raw_daily = out.loc[tested, "pair_valid"].fillna(False).astype(bool)
    raw_by_block = raw_daily.groupby(block_id).first()
    prior_passes = raw_by_block.shift(1).rolling(
        window=int(persistence_lookback),
        min_periods=int(persistence_lookback),
    ).sum()
    persistent_by_block = (
        raw_by_block
        & (prior_passes >= int(persistence_min_passes))
    )

    out["raw_pair_valid"] = out["pair_valid"].fillna(False).astype(bool)
    out["persistence_passes"] = np.nan
    out.loc[tested, "persistence_passes"] = block_id.map(prior_passes).astype(float).values
    out.loc[tested, "pair_valid"] = block_id.map(persistent_by_block).fillna(False).astype(bool).values
    return out



# ============================================================
# PAIR-BLOCK DIAGNOSTICS (PAST-ONLY MODEL STATE)
# ============================================================

def build_pair_block_diagnostics(model, industry, ticker_a, ticker_b):
    """Return one row per tested walk-forward block for research/export.

    Model-state fields are known at the start of the corresponding trading block.
    Future P&L, future Sharpe and future validity are deliberately excluded.
    """
    tested = model.loc[model["cointegration_pvalue"].notna()].copy()
    if tested.empty:
        return pd.DataFrame()

    comparison_columns = [
        "alpha", "beta", "cointegration_pvalue", "half_life", "orientation"
    ]
    sig = tested[comparison_columns].astype(str)
    starts = sig.ne(sig.shift()).any(axis=1)
    tested["_block_id"] = starts.cumsum().astype(int).values

    rows = []
    for block_id, g in tested.groupby("_block_id", sort=True):
        first = g.iloc[0]
        z = pd.to_numeric(g.get("zscore", pd.Series(index=g.index, dtype=float)), errors="coerce")
        rows.append({
            "industry": industry,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "pair": f"{ticker_a} / {ticker_b}",
            "block_number": int(block_id),
            "block_start": pd.Timestamp(g.index.min()),
            "block_end": pd.Timestamp(g.index.max()),
            "alpha": pd.to_numeric(pd.Series([first.get("alpha", np.nan)]), errors="coerce").iloc[0],
            "beta": pd.to_numeric(pd.Series([first.get("beta", np.nan)]), errors="coerce").iloc[0],
            "cointegration_pvalue": pd.to_numeric(pd.Series([first.get("cointegration_pvalue", np.nan)]), errors="coerce").iloc[0],
            "half_life": pd.to_numeric(pd.Series([first.get("half_life", np.nan)]), errors="coerce").iloc[0],
            "r_squared": pd.to_numeric(pd.Series([first.get("r_squared", np.nan)]), errors="coerce").iloc[0],
            "orientation": first.get("orientation", ""),
            "raw_pair_valid": bool(first.get("raw_pair_valid", first.get("pair_valid", False))),
            "persistence_passes": pd.to_numeric(pd.Series([first.get("persistence_passes", np.nan)]), errors="coerce").iloc[0],
            "pair_valid": bool(first.get("pair_valid", False)),
            "z_at_block_start": pd.to_numeric(pd.Series([first.get("zscore", np.nan)]), errors="coerce").iloc[0],
            "max_abs_z_in_block": float(z.abs().max()) if z.notna().any() else np.nan,
            "min_abs_z_in_block": float(z.abs().min()) if z.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)

# ============================================================
# ANALYSE ONE PAIR
# ============================================================

def analyse_pair(
    ticker_a,
    ticker_b,
    industry,
    opens,
    closes,
    formation_window=252,
    trading_window=21,
    z_window=60,
    coint_threshold=0.05,
    max_half_life=60,
    z_entry=2.0,
    z_exit=0.5,
    z_stop=3.5,
    gross_exposure=1.0,
    transaction_cost_bps=5.0,
    annual_borrow_bps=50.0,
    persistence_lookback=6,
    persistence_min_passes=3,
):

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    pair = prepare_pair_data(
        opens,
        closes,
        ticker_a,
        ticker_b,
    )

    minimum_required = (
        int(formation_window)
        + int(trading_window)
    )

    if len(pair) < minimum_required:

        return None

    # --------------------------------------------------------
    # WALK-FORWARD MODEL
    # --------------------------------------------------------

    model = build_walkforward(
        pair["close_a"],
        pair["close_b"],

        formation_window=int(
            formation_window
        ),

        trading_window=int(
            trading_window
        ),

        z_window=int(
            z_window
        ),

        coint_threshold=float(
            coint_threshold
        ),

        max_half_life=float(
            max_half_life
        ),
    )

    model = apply_persistence_filter(
        model,
        persistence_lookback=int(persistence_lookback),
        persistence_min_passes=int(persistence_min_passes),
    )

    model = model.join(
        pair[
            [
                "open_a",
                "open_b",
                "close_a",
                "close_b",
            ]
        ]
    )

    model_exec = model.dropna(
        subset=[
            "open_a",
            "open_b",
            "close_a",
            "close_b",
        ]
    )

    if model_exec.empty:

        return None

    # --------------------------------------------------------
    # WALK-FORWARD BLOCK DIAGNOSTICS
    # --------------------------------------------------------

    blocks = extract_blocks(
        model
    )

    block_diagnostics = build_pair_block_diagnostics(
        model, industry, ticker_a, ticker_b
    )

    total_blocks = len(
        blocks
    )

    if total_blocks:

        valid_blocks_df = blocks[
            blocks["pair_valid"]
        ].copy()

    else:

        valid_blocks_df = (
            pd.DataFrame()
        )

    valid_blocks = len(
        valid_blocks_df
    )

    # --------------------------------------------------------
    # RESEARCH FUNNEL DIAGNOSTICS
    # --------------------------------------------------------
    # All counts below are formation/trading-block counts and use only
    # information that was available at that block. They are diagnostic
    # only and never participate in pair selection.
    if total_blocks:
        coint_pass_blocks = int(
            (blocks["cointegration_pvalue"] < float(coint_threshold)).sum()
        )
        half_life_pass_blocks = int((
            (blocks["cointegration_pvalue"] < float(coint_threshold))
            & np.isfinite(blocks["half_life"])
            & (blocks["half_life"] > 0)
            & (blocks["half_life"] <= float(max_half_life))
        ).sum())
        persistence_pass_blocks = int(blocks["pair_valid"].fillna(False).sum())

        # Count eligible OOS blocks in which the spread actually reached
        # the entry threshold at least once. This is not a profitability
        # filter; it only measures whether a trade opportunity appeared.
        tested_mask = model["cointegration_pvalue"].notna()
        if tested_mask.any():
            sig_cols = ["alpha", "beta", "cointegration_pvalue", "half_life", "orientation"]
            sig = model.loc[tested_mask, sig_cols].astype(str)
            starts = sig.ne(sig.shift()).any(axis=1)
            daily_block_id = starts.cumsum()
            daily_diag = model.loc[tested_mask, ["pair_valid", "zscore"]].copy()
            daily_diag["block_id"] = daily_block_id.values
            daily_diag["entry_excursion"] = (
                daily_diag["pair_valid"].fillna(False)
                & daily_diag["zscore"].abs().ge(float(z_entry))
                & daily_diag["zscore"].abs().lt(float(z_stop))
            )
            z_entry_blocks = int(
                daily_diag.groupby("block_id")["entry_excursion"].any().sum()
            )
        else:
            z_entry_blocks = 0
    else:
        coint_pass_blocks = 0
        half_life_pass_blocks = 0
        persistence_pass_blocks = 0
        z_entry_blocks = 0

    if total_blocks > 0:

        valid_block_fraction = (
            valid_blocks
            / total_blocks
        )

    else:

        valid_block_fraction = 0.0

    # --------------------------------------------------------
    # FULL BLOCK STATISTICS
    # --------------------------------------------------------

    if total_blocks:

        median_pvalue = float(
            blocks[
                "cointegration_pvalue"
            ].median()
        )

        finite_half_life = (
            blocks.loc[
                np.isfinite(
                    blocks[
                        "half_life"
                    ]
                ),
                "half_life",
            ]
        )

        median_half_life = (
            float(
                finite_half_life.median()
            )
            if len(
                finite_half_life
            )
            else np.nan
        )

        median_r_squared = float(
            blocks[
                "r_squared"
            ].median()
        )

    else:

        median_pvalue = np.nan
        median_half_life = np.nan
        median_r_squared = np.nan

    # --------------------------------------------------------
    # VALID-BLOCK CONDITIONAL STATISTICS
    # --------------------------------------------------------

    if valid_blocks:

        valid_median_pvalue = float(
            valid_blocks_df[
                "cointegration_pvalue"
            ].median()
        )

        valid_half_life = (
            valid_blocks_df.loc[
                np.isfinite(
                    valid_blocks_df[
                        "half_life"
                    ]
                ),
                "half_life",
            ]
        )

        valid_median_half_life = (
            float(
                valid_half_life.median()
            )
            if len(
                valid_half_life
            )
            else np.nan
        )

        valid_median_r_squared = float(
            valid_blocks_df[
                "r_squared"
            ].median()
        )

    else:

        valid_median_pvalue = np.nan
        valid_median_half_life = np.nan
        valid_median_r_squared = np.nan

    # --------------------------------------------------------
    # DISTINCT VALIDITY REGIMES
    # --------------------------------------------------------

    validity_regimes = (
        count_validity_regimes(
            blocks[
                "pair_valid"
            ]
        )
        if total_blocks
        else 0
    )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    daily, ledger = run_execution(
        model_exec,

        gross_exposure=float(
            gross_exposure
        ),

        sizing_mode=(
            "Hedge-ratio weighted"
        ),

        transaction_cost_bps=float(
            transaction_cost_bps
        ),

        annual_borrow_bps=float(
            annual_borrow_bps
        ),

        z_entry=float(
            z_entry
        ),

        z_exit=float(
            z_exit
        ),

        z_stop=float(
            z_stop
        ),
    )

    perf = stats(
        daily["net_return"],
        daily["position"],
    )

    # --------------------------------------------------------
    # TRADE STATISTICS
    # --------------------------------------------------------

    trade_count = len(
        ledger
    )

    if trade_count:

        net_pnl = float(
            ledger[
                "net_pnl"
            ].sum()
        )

        gross_pnl = float(
            ledger[
                "gross_pnl"
            ].sum()
        )

        transaction_cost = float(
            ledger[
                "transaction_cost"
            ].sum()
        )

        borrow_cost = float(
            ledger[
                "borrow_cost"
            ].sum()
        )

        win_rate = float(
            (
                ledger[
                    "net_pnl"
                ]
                > 0
            ).mean()
        )

        average_holding_days = float(
            ledger[
                "holding_days"
            ].mean()
        )

        median_trade_pnl = float(
            ledger[
                "net_pnl"
            ].median()
        )

    else:

        net_pnl = 0.0
        gross_pnl = 0.0
        transaction_cost = 0.0
        borrow_cost = 0.0

        win_rate = np.nan
        average_holding_days = np.nan
        median_trade_pnl = np.nan

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "industry":
            industry,

        "ticker_a":
            ticker_a,

        "ticker_b":
            ticker_b,

        "pair":
            f"{ticker_a} / {ticker_b}",

        # Block diagnostics
        "total_blocks":
            total_blocks,

        "valid_blocks":
            valid_blocks,

        "valid_fraction":
            valid_block_fraction,

        "validity_regimes":
            validity_regimes,

        # Research funnel diagnostics (evaluation only)
        "coint_pass_blocks":
            coint_pass_blocks,

        "half_life_pass_blocks":
            half_life_pass_blocks,

        "persistence_pass_blocks":
            persistence_pass_blocks,

        "z_entry_blocks":
            z_entry_blocks,

        # Full-history formation diagnostics
        "median_cointegration_pvalue":
            median_pvalue,

        "median_half_life":
            median_half_life,

        "median_r_squared":
            median_r_squared,

        # Conditional diagnostics
        "valid_median_pvalue":
            valid_median_pvalue,

        "valid_median_half_life":
            valid_median_half_life,

        "valid_median_r_squared":
            valid_median_r_squared,

        # Trading diagnostics
        "trades":
            trade_count,

        "win_rate":
            win_rate,

        "avg_holding_days":
            average_holding_days,

        "median_trade_pnl":
            median_trade_pnl,

        # Performance
        "annual_return":
            perf["annual_return"],

        "annual_volatility":
            perf["annual_volatility"],

        "sharpe":
            perf["sharpe"],

        "max_drawdown":
            perf["max_drawdown"],

        "time_in_market":
            perf["time_in_market"],

        # P&L
        "gross_pnl":
            gross_pnl,

        "transaction_cost":
            transaction_cost,

        "borrow_cost":
            borrow_cost,

        "net_pnl":
            net_pnl,

        # Private payload consumed by portfolio_engine.py before the public
        # scanner DataFrame is built.
        "_block_diagnostics":
            block_diagnostics,
    }


# ============================================================
# RUN SELECTED INDUSTRY SCANNER
# ============================================================

def run_pair_scanner(
    industry, start, end, formation_window=252, trading_window=21, z_window=60,
    coint_threshold=0.05, max_half_life=60, z_entry=2.0, z_exit=0.5, z_stop=3.5,
    gross_exposure=1.0, transaction_cost_bps=5.0, annual_borrow_bps=50.0,
    use_prescreen=False, prescreen_max_pairs=None, prescreen_min_return_corr=None,
    prescreen_min_log_price_corr=None, persistence_lookback=6, persistence_min_passes=3,
):
    """Run every peer-group pair through the rolling formation/trading model.

    Correlation is not an eligibility screen. In each walk-forward block a pair is
    eligible only if the preceding formation window has Engle-Granger p below
    coint_threshold and a positive half-life no greater than max_half_life.
    """
    opens, closes = download_industry(industry, start, end)
    pairs = generate_industry_pairs(industry)

    results = []
    for ticker_a, ticker_b in pairs:
        try:
            result = analyse_pair(
                ticker_a=ticker_a, ticker_b=ticker_b, industry=industry,
                opens=opens, closes=closes, formation_window=formation_window,
                trading_window=trading_window, z_window=z_window,
                coint_threshold=coint_threshold, max_half_life=max_half_life,
                z_entry=z_entry, z_exit=z_exit, z_stop=z_stop,
                gross_exposure=gross_exposure, transaction_cost_bps=transaction_cost_bps,
                annual_borrow_bps=annual_borrow_bps,
                persistence_lookback=persistence_lookback,
                persistence_min_passes=persistence_min_passes,
            )
            if result is not None:
                result = dict(result)
                result.pop("_block_diagnostics", None)
                results.append(result)
        except Exception as exc:
            results.append({
                "industry": industry, "ticker_a": ticker_a, "ticker_b": ticker_b,
                "pair": f"{ticker_a} / {ticker_b}", "error": str(exc),
            })

    if not results:
        return pd.DataFrame()
    result_df = pd.DataFrame(results)
    if "valid_fraction" in result_df.columns:
        result_df = result_df.sort_values(
            ["valid_fraction", "valid_blocks", "validity_regimes", "trades"],
            ascending=[False, False, False, False], na_position="last",
        )
    return result_df.reset_index(drop=True)

