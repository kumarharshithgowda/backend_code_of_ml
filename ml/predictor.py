import sys
import pickle
import re
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Load models and data
try:
    with open('backend/ml/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('backend/ml/intent_classifier.pkl', 'rb') as f:
        intent_classifier = pickle.load(f)
    
    with open('backend/ml/stock_model.pkl', 'rb') as f:
        stock_model = pickle.load(f)
    
    with open('backend/ml/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('backend/ml/features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    with open('backend/ml/stock_info.pkl', 'rb') as f:
        stock_info = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    print("I'm having trouble accessing my knowledge base. Please try again later.")
    sys.exit(1)

# Extract stock ticker from user query
def extract_stock_ticker(query):
    # List of stock tickers we support
    supported_tickers = list(stock_info.keys())
    
    # Debug
    print(f"DEBUG: Query: {query}", file=sys.stderr)
    print(f"DEBUG: Supported tickers: {supported_tickers}", file=sys.stderr)
    
    # Check for exact matches of stock tickers in the query
    for ticker in supported_tickers:
        if ticker in query.upper().split():
            print(f"DEBUG: Found ticker: {ticker}", file=sys.stderr)
            return ticker
    
    # Check for partial matches (e.g., "apple" for "AAPL")
    ticker_names = {
        'apple': 'AAPL',
        'microsoft': 'MSFT',
        'google': 'GOOGL',
        'alphabet': 'GOOGL',
        'amazon': 'AMZN',
        'meta': 'META',
        'facebook': 'META',
        'fb': 'META',
        'tesla': 'TSLA',
        'nvidia': 'NVDA',
        'amd': 'AMD',
        'jpmorgan': 'JPM',
        'jp morgan': 'JPM',
        'chase': 'JPM',
        'visa': 'V',
        'walmart': 'WMT',
        'disney': 'DIS',
        'netflix': 'NFLX',
        'intel': 'INTC',
        'cisco': 'CSCO',
        'paypal': 'PYPL',
        'adobe': 'ADBE',
        'salesforce': 'CRM',
        'comcast': 'CMCSA',
        'pepsi': 'PEP',
        'pepsico': 'PEP',
        'costco': 'COST',
        'broadcom': 'AVGO',
        'texas instruments': 'TXN',
        'qualcomm': 'QCOM',
        'starbucks': 'SBUX'
    }
    
    query_lower = query.lower()
    for name, ticker in ticker_names.items():
        if name in query_lower:
            return ticker
    
    return None

# Get current stock price
def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            return f"The current price of {ticker} ({stock_info.get(ticker, '')}) is ${current_price:.2f}."
        else:
            return f"I couldn't retrieve the current price for {ticker}."
    except Exception as e:
        return f"I'm having trouble getting the price for {ticker} right now. Please try again later."

# Predict stock movement
def predict_stock_movement(ticker):
    try:
        # Get recent stock data
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Get 60 days of data for better analysis
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return f"I don't have enough data to make a prediction for {ticker}."
        
        # Get company info
        try:
            info = stock.info
            current_price = info.get('currentPrice', data['Close'].iloc[-1])
            market_cap = info.get('marketCap', 'N/A')
            if market_cap != 'N/A':
                market_cap = f"${market_cap/1000000000:.2f}B"
            pe_ratio = info.get('trailingPE', 'N/A')
            if pe_ratio != 'N/A':
                pe_ratio = f"{pe_ratio:.2f}"
            dividend_yield = info.get('dividendYield', 'N/A')
            if dividend_yield != 'N/A':
                dividend_yield = f"{dividend_yield*100:.2f}%"
        except:
            current_price = data['Close'].iloc[-1]
            market_cap = 'N/A'
            pe_ratio = 'N/A'
            dividend_yield = 'N/A'
        
        # Calculate features
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=min(50, len(data))).mean()
        data['MA200'] = data['Close'].rolling(window=min(200, len(data))).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=min(14, len(data))).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=min(14, len(data))).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['20d_std'] = data['Close'].rolling(window=min(20, len(data))).std()
        data['Upper_Band'] = data['MA20'] + (data['20d_std'] * 2)
        data['Lower_Band'] = data['MA20'] - (data['20d_std'] * 2)
        
        # Price change
        data['Price_Change'] = data['Close'].pct_change()
        
        # Volume analysis
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Volume_MA20'] = data['Volume'].rolling(window=min(20, len(data))).mean()
        volume_trend = "increasing" if data['Volume'].iloc[-5:].mean() > data['Volume_MA20'].iloc[-1] else "decreasing"
        
        # Trend analysis
        short_term_trend = "upward" if data['Close'].iloc[-1] > data['MA20'].iloc[-1] else "downward"
        medium_term_trend = "upward" if data['Close'].iloc[-1] > data['MA50'].iloc[-1] else "downward"
        
        # Support and resistance levels
        recent_data = data.iloc[-30:]
        support_level = recent_data['Low'].min()
        resistance_level = recent_data['High'].max()
        
        # Prepare feature vector for the most recent data point
        latest_data = data.iloc[-1].copy()
        
        # Create a DataFrame with one row for prediction
        pred_data = pd.DataFrame({
            'Price_Change': [latest_data['Price_Change']]
        })
        
        # Add dummy columns for the stock
        for feature in features:
            if feature.startswith('stock_'):
                stock_ticker = feature.split('_')[1]
                if stock_ticker == ticker:
                    pred_data[feature] = 1
                else:
                    pred_data[feature] = 0
            elif feature != 'Price_Change' and feature in latest_data:
                pred_data[feature] = latest_data[feature]
            elif feature != 'Price_Change':
                pred_data[feature] = 0  # Default value for missing features
        
        # Ensure all features are present
        for feature in features:
            if feature not in pred_data.columns:
                pred_data[feature] = 0
        
        # Scale the features
        X = pred_data[features]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = stock_model.predict(X_scaled)[0]
        probability = stock_model.predict_proba(X_scaled)[0]
        confidence = max(probability) * 100
        
        # Technical indicators analysis
        rsi_signal = "oversold" if latest_data['RSI'] < 30 else "overbought" if latest_data['RSI'] > 70 else "neutral"
        macd_signal = "bullish" if latest_data['MACD'] > latest_data['Signal_Line'] else "bearish"
        
        # Bollinger Bands analysis
        bb_position = (latest_data['Close'] - latest_data['Lower_Band']) / (latest_data['Upper_Band'] - latest_data['Lower_Band'])
        bb_signal = "oversold" if bb_position < 0.2 else "overbought" if bb_position > 0.8 else "neutral"
        
        # Calculate recent performance
        week_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100 if len(data) >= 5 else 0
        month_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
        
        # Compile the prediction result
        if prediction == 1:
            result = f"<strong>📈 {ticker} ({stock_info.get(ticker, '')}) Prediction: UPWARD</strong> with {confidence:.1f}% confidence"
        else:
            result = f"<strong>📉 {ticker} ({stock_info.get(ticker, '')}) Prediction: DOWNWARD</strong> with {confidence:.1f}% confidence"
        
        # Add company info
        result += f"<br><br><strong>Current Price:</strong> ${current_price:.2f}"
        result += f"<br><strong>Market Cap:</strong> {market_cap}"
        result += f"<br><strong>P/E Ratio:</strong> {pe_ratio}"
        result += f"<br><strong>Dividend Yield:</strong> {dividend_yield}"
        
        # Add performance metrics
        result += f"<br><br><strong>Recent Performance:</strong>"
        result += f"<br>• 1 Week: {week_change:.2f}%"
        result += f"<br>• 1 Month: {month_change:.2f}%"
        
        # Add technical analysis
        result += f"<br><br><strong>Technical Analysis:</strong>"
        result += f"<br>• RSI: {latest_data['RSI']:.2f} ({rsi_signal})"
        result += f"<br>• MACD: {macd_signal}"
        result += f"<br>• Bollinger Bands: {bb_signal}"
        result += f"<br>• Volume Trend: {volume_trend}"
        result += f"<br>• Short-term Trend: {short_term_trend}"
        result += f"<br>• Medium-term Trend: {medium_term_trend}"
        
        # Add support and resistance levels
        result += f"<br><br><strong>Key Levels:</strong>"
        result += f"<br>• Support: ${support_level:.2f}"
        result += f"<br>• Resistance: ${resistance_level:.2f}"
        
        # Add recommendation
        if prediction == 1 and confidence > 70:
            result += f"<br><br><strong>Recommendation:</strong> Consider buying if it aligns with your investment strategy."
        elif prediction == 0 and confidence > 70:
            result += f"<br><br><strong>Recommendation:</strong> Consider selling or avoiding new positions."
        else:
            result += f"<br><br><strong>Recommendation:</strong> Monitor closely as the signal is not strong enough for a definitive recommendation."
        
        # Add disclaimer
        result += f"<br><br><em>Disclaimer: This is not financial advice. Always do your own research before making investment decisions.</em>"
        
        return result
    except Exception as e:
        return f"I'm having trouble making a prediction for {ticker} right now. Please try again later. Error: {str(e)}"

# Get market sentiment
def get_market_sentiment():
    try:
        # Get data for major indices
        indices = {
            'S&P 500': 'SPY',
            'Dow Jones': 'DIA',
            'Nasdaq': 'QQQ',
            'Russell 2000': 'IWM'
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        results = {}
        overall_sentiment = 0
        
        for index_name, ticker in indices.items():
            index = yf.Ticker(ticker)
            data = index.history(start=start_date, end=end_date)
            
            if data.empty:
                continue
                
            # Calculate performance metrics
            month_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            week_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100 if len(data) >= 5 else 0
            day_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100 if len(data) >= 2 else 0
            
            # Calculate technical indicators
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # MACD
            ema12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal_line
            
            # Determine sentiment for this index
            if month_change > 5:
                index_sentiment = 2  # strongly bullish
                sentiment_text = "strongly bullish"
            elif month_change > 2:
                index_sentiment = 1  # bullish
                sentiment_text = "bullish"
            elif month_change > -2:
                index_sentiment = 0  # neutral
                sentiment_text = "neutral"
            elif month_change > -5:
                index_sentiment = -1  # bearish
                sentiment_text = "bearish"
            else:
                index_sentiment = -2  # strongly bearish
                sentiment_text = "strongly bearish"
                
            overall_sentiment += index_sentiment
            
            # Store results
            results[index_name] = {
                'month_change': month_change,
                'week_change': week_change,
                'day_change': day_change,
                'current_price': data['Close'].iloc[-1],
                'rsi': current_rsi,
                'macd_histogram': macd_histogram.iloc[-1],
                'sentiment': sentiment_text
            }
        
        if not results:
            return "I don't have enough data to analyze market sentiment right now."
        
        # Calculate overall market sentiment
        overall_sentiment = overall_sentiment / len(results)
        if overall_sentiment > 1.5:
            market_sentiment = "strongly bullish"
        elif overall_sentiment > 0.5:
            market_sentiment = "bullish"
        elif overall_sentiment > -0.5:
            market_sentiment = "neutral"
        elif overall_sentiment > -1.5:
            market_sentiment = "bearish"
        else:
            market_sentiment = "strongly bearish"
        
        # Generate response
        response = f"<strong>📊 Current Market Sentiment: {market_sentiment.upper()}</strong><br><br>"
        
        # Add index performance table
        response += "<strong>Major Indices Performance:</strong><br>"
        for index_name, data in results.items():
            response += f"<br>• <strong>{index_name}:</strong> ${data['current_price']:.2f} "
            
            # Add arrows for daily change
            if data['day_change'] > 0:
                response += f"<span style='color:green'>▲ {data['day_change']:.2f}%</span> today"
            else:
                response += f"<span style='color:red'>▼ {data['day_change']:.2f}%</span> today"
                
            response += f" | {data['month_change']:.2f}% in 30 days"
            response += f" | RSI: {data['rsi']:.1f}"
            response += f" | Sentiment: {data['sentiment']}"
        
        # Add market analysis
        response += "<br><br><strong>Market Analysis:</strong><br>"
        
        # Analyze RSI across indices
        avg_rsi = sum(data['rsi'] for data in results.values()) / len(results)
        if avg_rsi > 70:
            response += "<br>• The market appears <strong>overbought</strong> based on RSI indicators, suggesting potential for a pullback."
        elif avg_rsi < 30:
            response += "<br>• The market appears <strong>oversold</strong> based on RSI indicators, suggesting potential for a bounce."
        else:
            response += f"<br>• RSI indicators are in a <strong>neutral zone</strong> at {avg_rsi:.1f}, neither overbought nor oversold."
        
        # Analyze MACD across indices
        macd_positive = sum(1 for data in results.values() if data['macd_histogram'] > 0)
        macd_negative = len(results) - macd_positive
        
        if macd_positive > macd_negative:
            response += "<br>• MACD indicators are showing <strong>bullish momentum</strong> across most major indices."
        else:
            response += "<br>• MACD indicators are showing <strong>bearish momentum</strong> across most major indices."
        
        # Add sector performance if available
        try:
            sectors = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Consumer Discretionary': 'XLY',
                'Utilities': 'XLU'
            }
            
            sector_performance = {}
            for sector_name, ticker in sectors.items():
                sector = yf.Ticker(ticker)
                data = sector.history(start=start_date, end=end_date)
                if not data.empty:
                    month_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    sector_performance[sector_name] = month_change
            
            if sector_performance:
                # Sort sectors by performance
                sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
                
                response += "<br><br><strong>Sector Performance (30 days):</strong><br>"
                for sector_name, change in sorted_sectors:
                    if change > 0:
                        response += f"<br>• {sector_name}: <span style='color:green'>▲ {change:.2f}%</span>"
                    else:
                        response += f"<br>• {sector_name}: <span style='color:red'>▼ {change:.2f}%</span>"
        except:
            # If sector analysis fails, continue without it
            pass
        
        # Add market outlook
        response += "<br><br><strong>Market Outlook:</strong><br>"
        if market_sentiment == "strongly bullish":
            response += "<br>• The market is showing strong positive momentum across major indices."
            response += "<br>• This suggests a favorable environment for growth stocks and risk assets."
        elif market_sentiment == "bullish":
            response += "<br>• The market is trending upward with positive momentum."
            response += "<br>• This generally indicates favorable conditions for equity investments."
        elif market_sentiment == "neutral":
            response += "<br>• The market is showing mixed signals without a clear directional bias."
            response += "<br>• This suggests a cautious approach and sector-specific analysis may be more valuable."
        elif market_sentiment == "bearish":
            response += "<br>• The market is trending downward with negative momentum."
            response += "<br>• This may indicate a challenging environment for growth stocks."
        else:  # strongly bearish
            response += "<br>• The market is showing strong negative momentum across major indices."
            response += "<br>• This suggests a defensive positioning may be prudent."
        
        # Add disclaimer
        response += "<br><br><em>Disclaimer: This analysis is based on technical indicators and recent price action. Market conditions can change rapidly and this should not be considered financial advice.</em>"
        
        return response
    except Exception as e:
        return f"I'm having trouble analyzing market sentiment right now. Please try again later. Error: {str(e)}"

# Get stock news
def get_stock_news(ticker):
    try:
        # Get company info and news
        stock = yf.Ticker(ticker)
        
        # Try to get company information
        try:
            info = stock.info
            company_name = info.get('longName', stock_info.get(ticker, ticker))
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            business_summary = info.get('longBusinessSummary', 'No business summary available.')
        except:
            company_name = stock_info.get(ticker, ticker)
            sector = 'N/A'
            industry = 'N/A'
            business_summary = 'No business summary available.'
        
        # Format the response
        response = f"<strong>📰 {company_name} ({ticker}) News & Information</strong><br><br>"
        
        # Add company information
        response += "<strong>Company Profile:</strong><br>"
        response += f"• <strong>Sector:</strong> {sector}<br>"
        response += f"• <strong>Industry:</strong> {industry}<br><br>"
        
        # Add business summary (truncate if too long)
        if len(business_summary) > 500:
            business_summary = business_summary[:500] + "..."
        response += f"<strong>Business Summary:</strong><br>{business_summary}<br><br>"
        
        # Add news sources
        response += "<strong>Recommended News Sources:</strong><br>"
        response += f"• <a href='https://finance.yahoo.com/quote/{ticker}' target='_blank'>Yahoo Finance</a> - Comprehensive financial data and news<br>"
        response += f"• <a href='https://www.cnbc.com/quotes/{ticker}' target='_blank'>CNBC</a> - Breaking news and market analysis<br>"
        response += f"• <a href='https://www.bloomberg.com/quote/{ticker}:US' target='_blank'>Bloomberg</a> - In-depth financial reporting<br>"
        response += f"• <a href='https://seekingalpha.com/symbol/{ticker}' target='_blank'>Seeking Alpha</a> - Investor perspectives and analysis<br>"
        response += f"• <a href='https://www.marketwatch.com/investing/stock/{ticker}' target='_blank'>MarketWatch</a> - Market data and business news<br><br>"
        
        # Add recent events that might affect the stock
        response += "<strong>Key Factors to Monitor:</strong><br>"
        
        # Add industry-specific factors based on sector
        if sector == 'Technology':
            response += "• Product launches and innovation pipeline<br>"
            response += "• Competitive positioning in the tech landscape<br>"
            response += "• AI and cloud computing initiatives<br>"
        elif sector == 'Healthcare':
            response += "• FDA approvals and clinical trial results<br>"
            response += "• Healthcare policy changes<br>"
            response += "• Patent expirations and new drug pipelines<br>"
        elif sector == 'Financial Services':
            response += "• Interest rate changes and Fed policy<br>"
            response += "• Loan performance and credit quality<br>"
            response += "• Regulatory developments<br>"
        elif sector == 'Consumer Cyclical':
            response += "• Consumer spending trends<br>"
            response += "• E-commerce performance<br>"
            response += "• Brand strength and market share<br>"
        elif sector == 'Energy':
            response += "• Oil and gas price movements<br>"
            response += "• Renewable energy initiatives<br>"
            response += "• Regulatory and environmental policies<br>"
        else:
            response += "• Quarterly earnings reports and guidance<br>"
            response += "• Management changes and strategic initiatives<br>"
            response += "• Industry trends and competitive landscape<br>"
        
        # Add general factors
        response += "• Macroeconomic indicators and market sentiment<br>"
        response += "• Analyst upgrades/downgrades and price targets<br>"
        
        # Add disclaimer
        response += "<br><em>Disclaimer: This information is provided for educational purposes only and should not be considered investment advice.</em>"
        
        return response
    except Exception as e:
        return f"For the latest news on {ticker} ({stock_info.get(ticker, '')}), I recommend checking financial news websites like Yahoo Finance, CNBC, or Bloomberg. Error retrieving detailed information: {str(e)}"

# Process the user query
def process_query(query):
    query_lower = query.lower()
    
    # Check for special queries first
    if "compare" in query_lower and "and" in query_lower:
        # Handle stock comparison queries
        tickers = []
        for ticker in stock_info.keys():
            if ticker in query.upper().split() or ticker.lower() in query_lower:
                tickers.append(ticker)
        
        if len(tickers) >= 2:
            return compare_stocks(tickers[0], tickers[1])
        else:
            return "Please specify two valid stock tickers to compare, for example: 'Compare AAPL and MSFT'"
    
    elif "sector" in query_lower or "sectors" in query_lower:
        # Handle sector analysis queries
        return analyze_sectors()
    
    elif "portfolio" in query_lower:
        # Handle portfolio queries
        tickers = []
        for ticker in stock_info.keys():
            if ticker in query.upper().split() or ticker.lower() in query_lower:
                tickers.append(ticker)
        
        if tickers:
            return analyze_portfolio(tickers)
        else:
            return "Please specify which stocks you'd like to include in your portfolio analysis."
    
    # Standard intent classification
    X = vectorizer.transform([query])
    intent = intent_classifier.predict(X)[0]
    
    # Extract stock ticker if present
    ticker = extract_stock_ticker(query)
    
    # Handle different intents
    if intent == "stock_price":
        if ticker:
            return get_stock_price(ticker)
        else:
            return "Which stock would you like to know the price of? Please specify a stock ticker (e.g., AAPL, MSFT, GOOGL)."
    
    elif intent == "prediction":
        if ticker:
            return predict_stock_movement(ticker)
        else:
            return "Which stock would you like me to analyze? Please specify a stock ticker (e.g., AAPL, MSFT, GOOGL)."
    
    elif intent == "market_sentiment":
        return get_market_sentiment()
    
    elif intent == "stock_news":
        if ticker:
            return get_stock_news(ticker)
        else:
            return "Which stock would you like news about? Please specify a stock ticker (e.g., AAPL, MSFT, GOOGL)."
    
    # If we have a ticker but unclear intent, provide comprehensive info
    elif ticker:
        if "fundamentals" in query_lower or "pe" in query_lower or "ratio" in query_lower:
            return get_stock_fundamentals(ticker)
        else:
            return predict_stock_movement(ticker)
    
    else:
        return "I'm not sure I understand. You can ask me about stock prices, predictions, market sentiment, or news. You can also compare stocks, analyze sectors, or check stock fundamentals."

# Compare two stocks
def compare_stocks(ticker1, ticker2):
    try:
        # Get data for both stocks
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data1 = stock1.history(start=start_date, end=end_date)
        data2 = stock2.history(start=start_date, end=end_date)
        
        if data1.empty or data2.empty:
            return f"I don't have enough data to compare {ticker1} and {ticker2}."
        
        # Get company info
        try:
            info1 = stock1.info
            info2 = stock2.info
            
            name1 = info1.get('longName', stock_info.get(ticker1, ticker1))
            name2 = info2.get('longName', stock_info.get(ticker2, ticker2))
            
            price1 = info1.get('currentPrice', data1['Close'].iloc[-1])
            price2 = info2.get('currentPrice', data2['Close'].iloc[-1])
            
            market_cap1 = info1.get('marketCap', 'N/A')
            market_cap2 = info2.get('marketCap', 'N/A')
            
            if market_cap1 != 'N/A':
                market_cap1 = f"${market_cap1/1000000000:.2f}B"
            if market_cap2 != 'N/A':
                market_cap2 = f"${market_cap2/1000000000:.2f}B"
            
            pe1 = info1.get('trailingPE', 'N/A')
            pe2 = info2.get('trailingPE', 'N/A')
            
            if pe1 != 'N/A':
                pe1 = f"{pe1:.2f}"
            if pe2 != 'N/A':
                pe2 = f"{pe2:.2f}"
            
            div_yield1 = info1.get('dividendYield', 'N/A')
            div_yield2 = info2.get('dividendYield', 'N/A')
            
            if div_yield1 != 'N/A':
                div_yield1 = f"{div_yield1*100:.2f}%"
            if div_yield2 != 'N/A':
                div_yield2 = f"{div_yield2*100:.2f}%"
            
        except:
            name1 = stock_info.get(ticker1, ticker1)
            name2 = stock_info.get(ticker2, ticker2)
            price1 = data1['Close'].iloc[-1]
            price2 = data2['Close'].iloc[-1]
            market_cap1 = 'N/A'
            market_cap2 = 'N/A'
            pe1 = 'N/A'
            pe2 = 'N/A'
            div_yield1 = 'N/A'
            div_yield2 = 'N/A'
        
        # Calculate performance metrics
        month_change1 = ((data1['Close'].iloc[-1] / data1['Close'].iloc[0]) - 1) * 100
        month_change2 = ((data2['Close'].iloc[-1] / data2['Close'].iloc[0]) - 1) * 100
        
        # Calculate technical indicators
        # RSI
        delta1 = data1['Close'].diff()
        gain1 = delta1.where(delta1 > 0, 0).rolling(window=14).mean()
        loss1 = -delta1.where(delta1 < 0, 0).rolling(window=14).mean()
        rs1 = gain1 / loss1
        rsi1 = 100 - (100 / (1 + rs1))
        
        delta2 = data2['Close'].diff()
        gain2 = delta2.where(delta2 > 0, 0).rolling(window=14).mean()
        loss2 = -delta2.where(delta2 < 0, 0).rolling(window=14).mean()
        rs2 = gain2 / loss2
        rsi2 = 100 - (100 / (1 + rs2))
        
        # Format the response
        response = f"<strong>📊 Stock Comparison: {name1} vs {name2}</strong><br><br>"
        
        # Create comparison table
        response += "<table style='width:100%; border-collapse: collapse;'>"
        
        # Header row
        response += "<tr style='background-color:#f2f2f2;'>"
        response += "<th style='padding:8px; text-align:left;'>Metric</th>"
        response += f"<th style='padding:8px; text-align:right;'>{ticker1}</th>"
        response += f"<th style='padding:8px; text-align:right;'>{ticker2}</th>"
        response += "</tr>"
        
        # Current Price
        response += "<tr>"
        response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Current Price</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>${price1:.2f}</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>${price2:.2f}</td>"
        response += "</tr>"
        
        # 30-Day Performance
        response += "<tr>"
        response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>30-Day Performance</td>"
        
        # Add color coding for performance
        color1 = "green" if month_change1 > 0 else "red"
        color2 = "green" if month_change2 > 0 else "red"
        
        response += f"<td style='padding:8px; text-align:right; color:{color1}; border-bottom:1px solid #ddd;'>{month_change1:.2f}%</td>"
        response += f"<td style='padding:8px; text-align:right; color:{color2}; border-bottom:1px solid #ddd;'>{month_change2:.2f}%</td>"
        response += "</tr>"
        
        # Market Cap
        response += "<tr>"
        response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Market Cap</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{market_cap1}</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{market_cap2}</td>"
        response += "</tr>"
        
        # P/E Ratio
        response += "<tr>"
        response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>P/E Ratio</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{pe1}</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{pe2}</td>"
        response += "</tr>"
        
        # Dividend Yield
        response += "<tr>"
        response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Dividend Yield</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{div_yield1}</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{div_yield2}</td>"
        response += "</tr>"
        
        # RSI
        response += "<tr>"
        response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>RSI (14-day)</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{rsi1.iloc[-1]:.2f}</td>"
        response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{rsi2.iloc[-1]:.2f}</td>"
        response += "</tr>"
        
        response += "</table>"
        
        # Add analysis
        response += "<br><strong>Analysis:</strong><br>"
        
        # Performance comparison
        if month_change1 > month_change2:
            perf_diff = month_change1 - month_change2
            response += f"• {ticker1} has outperformed {ticker2} by {perf_diff:.2f}% over the past month.<br>"
        elif month_change2 > month_change1:
            perf_diff = month_change2 - month_change1
            response += f"• {ticker2} has outperformed {ticker1} by {perf_diff:.2f}% over the past month.<br>"
        else:
            response += f"• Both stocks have shown similar performance over the past month.<br>"
        
        # Valuation comparison if P/E ratios are available
        if pe1 != 'N/A' and pe2 != 'N/A':
            pe1_float = float(pe1)
            pe2_float = float(pe2)
            if pe1_float < pe2_float:
                response += f"• {ticker1} has a lower P/E ratio, which might indicate it's more attractively valued compared to {ticker2}.<br>"
            elif pe2_float < pe1_float:
                response += f"• {ticker2} has a lower P/E ratio, which might indicate it's more attractively valued compared to {ticker1}.<br>"
        
        # RSI comparison
        rsi1_val = rsi1.iloc[-1]
        rsi2_val = rsi2.iloc[-1]
        
        if rsi1_val > 70 and rsi2_val > 70:
            response += f"• Both stocks are currently showing overbought conditions based on RSI.<br>"
        elif rsi1_val < 30 and rsi2_val < 30:
            response += f"• Both stocks are currently showing oversold conditions based on RSI.<br>"
        elif rsi1_val > 70:
            response += f"• {ticker1} is showing overbought conditions based on RSI.<br>"
        elif rsi2_val > 70:
            response += f"• {ticker2} is showing overbought conditions based on RSI.<br>"
        elif rsi1_val < 30:
            response += f"• {ticker1} is showing oversold conditions based on RSI.<br>"
        elif rsi2_val < 30:
            response += f"• {ticker2} is showing oversold conditions based on RSI.<br>"
        
        # Add disclaimer
        response += "<br><em>Disclaimer: This comparison is for informational purposes only and should not be considered investment advice.</em>"
        
        return response
    except Exception as e:
        return f"I'm having trouble comparing {ticker1} and {ticker2} right now. Please try again later. Error: {str(e)}"

# Analyze market sectors
def analyze_sectors():
    try:
        # Define sector ETFs
        sectors = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Utilities': 'XLU',
            'Materials': 'XLB',
            'Industrials': 'XLI',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC'
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        sector_data = {}
        
        for sector_name, ticker in sectors.items():
            etf = yf.Ticker(ticker)
            data = etf.history(start=start_date, end=end_date)
            
            if not data.empty:
                # Calculate performance metrics
                month_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                week_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100 if len(data) >= 5 else 0
                
                # Calculate RSI
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                sector_data[sector_name] = {
                    'month_change': month_change,
                    'week_change': week_change,
                    'current_price': data['Close'].iloc[-1],
                    'rsi': rsi.iloc[-1]
                }
        
        if not sector_data:
            return "I don't have enough data to analyze market sectors right now."
        
        # Sort sectors by monthly performance
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]['month_change'], reverse=True)
        
        # Format the response
        response = "<strong>📊 Sector Performance Analysis</strong><br><br>"
        
        # Create a table for sector performance
        response += "<table style='width:100%; border-collapse: collapse;'>"
        
        # Header row
        response += "<tr style='background-color:#f2f2f2;'>"
        response += "<th style='padding:8px; text-align:left;'>Sector</th>"
        response += "<th style='padding:8px; text-align:right;'>1-Month Return</th>"
        response += "<th style='padding:8px; text-align:right;'>1-Week Return</th>"
        response += "<th style='padding:8px; text-align:right;'>RSI</th>"
        response += "</tr>"
        
        # Add rows for each sector
        for sector_name, data in sorted_sectors:
            month_change = data['month_change']
            week_change = data['week_change']
            rsi = data['rsi']
            
            # Determine colors based on performance
            month_color = "green" if month_change > 0 else "red"
            week_color = "green" if week_change > 0 else "red"
            
            response += "<tr>"
            response += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{sector_name}</td>"
            response += f"<td style='padding:8px; text-align:right; color:{month_color}; border-bottom:1px solid #ddd;'>{month_change:.2f}%</td>"
            response += f"<td style='padding:8px; text-align:right; color:{week_color}; border-bottom:1px solid #ddd;'>{week_change:.2f}%</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{rsi:.2f}</td>"
            response += "</tr>"
        
        response += "</table>"
        
        # Add analysis
        response += "<br><strong>Analysis:</strong><br>"
        
        # Top and bottom performers
        top_sector = sorted_sectors[0][0]
        bottom_sector = sorted_sectors[-1][0]
        
        response += f"• <strong>{top_sector}</strong> has been the best performing sector over the past month with a return of {sorted_sectors[0][1]['month_change']:.2f}%.<br>"
        response += f"• <strong>{bottom_sector}</strong> has been the worst performing sector with a return of {sorted_sectors[-1][1]['month_change']:.2f}%.<br>"
        
        # Identify overbought/oversold sectors
        overbought_sectors = [sector for sector, data in sorted_sectors if data['rsi'] > 70]
        oversold_sectors = [sector for sector, data in sorted_sectors if data['rsi'] < 30]
        
        if overbought_sectors:
            response += f"• The following sectors appear overbought based on RSI: {', '.join(overbought_sectors)}.<br>"
        
        if oversold_sectors:
            response += f"• The following sectors appear oversold based on RSI: {', '.join(oversold_sectors)}.<br>"
        
        # Market rotation analysis
        if sorted_sectors[0][1]['week_change'] > sorted_sectors[0][1]['month_change'] / 4:  # If weekly return is stronger than would be expected
            response += f"• There appears to be accelerating momentum in the {top_sector} sector.<br>"
        
        # Defensive vs. cyclical analysis
        defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
        cyclical_sectors = ['Technology', 'Consumer Discretionary', 'Industrials', 'Financials']
        
        defensive_perf = [data['month_change'] for sector, data in sorted_sectors if sector in defensive_sectors]
        cyclical_perf = [data['month_change'] for sector, data in sorted_sectors if sector in cyclical_sectors]
        
        if defensive_perf and cyclical_perf:  # Make sure we have data for both
            avg_defensive = sum(defensive_perf) / len(defensive_perf)
            avg_cyclical = sum(cyclical_perf) / len(cyclical_perf)
            
            if avg_cyclical > avg_defensive:
                response += "• Cyclical sectors are outperforming defensive sectors, which typically indicates risk-on market sentiment.<br>"
            else:
                response += "• Defensive sectors are outperforming cyclical sectors, which may indicate cautious market sentiment.<br>"
        
        # Add disclaimer
        response += "<br><em>Disclaimer: Past performance is not indicative of future results. This analysis is for informational purposes only.</em>"
        
        return response
    except Exception as e:
        return f"I'm having trouble analyzing market sectors right now. Please try again later. Error: {str(e)}"

# Get stock fundamentals
def get_stock_fundamentals(ticker):
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        
        try:
            info = stock.info
            company_name = info.get('longName', stock_info.get(ticker, ticker))
            
            # Financial metrics
            market_cap = info.get('marketCap', 'N/A')
            if market_cap != 'N/A':
                market_cap = f"${market_cap/1000000000:.2f}B"
                
            pe_ratio = info.get('trailingPE', 'N/A')
            if pe_ratio != 'N/A':
                pe_ratio = f"{pe_ratio:.2f}"
                
            forward_pe = info.get('forwardPE', 'N/A')
            if forward_pe != 'N/A':
                forward_pe = f"{forward_pe:.2f}"
                
            peg_ratio = info.get('pegRatio', 'N/A')
            if peg_ratio != 'N/A':
                peg_ratio = f"{peg_ratio:.2f}"
                
            price_to_book = info.get('priceToBook', 'N/A')
            if price_to_book != 'N/A':
                price_to_book = f"{price_to_book:.2f}"
                
            dividend_yield = info.get('dividendYield', 'N/A')
            if dividend_yield != 'N/A':
                dividend_yield = f"{dividend_yield*100:.2f}%"
                
            profit_margins = info.get('profitMargins', 'N/A')
            if profit_margins != 'N/A':
                profit_margins = f"{profit_margins*100:.2f}%"
                
            return_on_equity = info.get('returnOnEquity', 'N/A')
            if return_on_equity != 'N/A':
                return_on_equity = f"{return_on_equity*100:.2f}%"
                
            revenue_growth = info.get('revenueGrowth', 'N/A')
            if revenue_growth != 'N/A':
                revenue_growth = f"{revenue_growth*100:.2f}%"
                
            debt_to_equity = info.get('debtToEquity', 'N/A')
            if debt_to_equity != 'N/A':
                debt_to_equity = f"{debt_to_equity:.2f}"
                
            current_price = info.get('currentPrice', 'N/A')
            if current_price != 'N/A':
                current_price = f"${current_price:.2f}"
                
            target_price = info.get('targetMeanPrice', 'N/A')
            if target_price != 'N/A':
                target_price = f"${target_price:.2f}"
                
            # Format the response
            response = f"<strong>📈 {company_name} ({ticker}) Fundamentals</strong><br><br>"
            
            # Current price and target
            response += f"<strong>Current Price:</strong> {current_price}<br>"
            if target_price != 'N/A':
                response += f"<strong>Analyst Target:</strong> {target_price}<br><br>"
            else:
                response += "<br>"
            
            # Create a table for fundamentals
            response += "<table style='width:100%; border-collapse: collapse;'>"
            
            # Valuation metrics
            response += "<tr style='background-color:#f2f2f2;'>"
            response += "<th colspan='2' style='padding:8px; text-align:left;'>Valuation Metrics</th>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Market Cap</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{market_cap}</td>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>P/E Ratio (TTM)</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{pe_ratio}</td>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Forward P/E</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{forward_pe}</td>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>PEG Ratio</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{peg_ratio}</td>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Price/Book</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{price_to_book}</td>"
            response += "</tr>"
            
            # Profitability metrics
            response += "<tr style='background-color:#f2f2f2;'>"
            response += "<th colspan='2' style='padding:8px; text-align:left;'>Profitability & Growth</th>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Profit Margin</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{profit_margins}</td>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Return on Equity</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{return_on_equity}</td>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Revenue Growth</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{revenue_growth}</td>"
            response += "</tr>"
            
            # Financial health
            response += "<tr style='background-color:#f2f2f2;'>"
            response += "<th colspan='2' style='padding:8px; text-align:left;'>Financial Health</th>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Debt to Equity</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{debt_to_equity}</td>"
            response += "</tr>"
            
            response += "<tr>"
            response += "<td style='padding:8px; border-bottom:1px solid #ddd;'>Dividend Yield</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>{dividend_yield}</td>"
            response += "</tr>"
            
            response += "</table>"
            
            # Add analysis
            response += "<br><strong>Analysis:</strong><br>"
            
            # P/E ratio analysis
            if pe_ratio != 'N/A' and forward_pe != 'N/A':
                pe_float = float(pe_ratio)
                forward_pe_float = float(forward_pe)
                
                if pe_float < 15:
                    response += f"• {ticker} has a relatively low P/E ratio of {pe_ratio}, which may indicate it's undervalued compared to the broader market.<br>"
                elif pe_float > 30:
                    response += f"• {ticker} has a relatively high P/E ratio of {pe_ratio}, which may indicate higher growth expectations or potential overvaluation.<br>"
                
                if forward_pe_float < pe_float:
                    response += f"• The forward P/E of {forward_pe} is lower than the trailing P/E, suggesting analysts expect earnings to improve.<br>"
                else:
                    response += f"• The forward P/E of {forward_pe} is higher than the trailing P/E, suggesting analysts expect earnings to decline.<br>"
            
            # Dividend analysis
            if dividend_yield != 'N/A':
                dividend_float = float(dividend_yield.replace('%', ''))
                if dividend_float > 3:
                    response += f"• {ticker} offers an above-average dividend yield of {dividend_yield}.<br>"
                elif dividend_float == 0:
                    response += f"• {ticker} does not currently pay a dividend.<br>"
            
            # Debt analysis
            if debt_to_equity != 'N/A':
                debt_float = float(debt_to_equity)
                if debt_float > 2:
                    response += f"• The debt-to-equity ratio of {debt_to_equity} is relatively high, which may indicate higher financial risk.<br>"
                elif debt_float < 0.5:
                    response += f"• The debt-to-equity ratio of {debt_to_equity} is relatively low, suggesting a strong balance sheet.<br>"
            
            # Add disclaimer
            response += "<br><em>Disclaimer: This fundamental analysis is for informational purposes only and should not be considered investment advice.</em>"
            
            return response
        except Exception as e:
            return f"I'm having trouble retrieving fundamental data for {ticker}. Error: {str(e)}"
    except Exception as e:
        return f"I'm having trouble analyzing {ticker} fundamentals right now. Please try again later. Error: {str(e)}"

# Analyze a portfolio of stocks
def analyze_portfolio(tickers):
    try:
        if len(tickers) > 10:
            tickers = tickers[:10]  # Limit to 10 stocks for performance
            
        response = f"<strong>📊 Portfolio Analysis ({len(tickers)} stocks)</strong><br><br>"
        
        # Get data for all stocks
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        portfolio_data = {}
        total_performance = 0
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                continue
                
            # Get company info
            try:
                info = stock.info
                company_name = info.get('longName', stock_info.get(ticker, ticker))
                current_price = info.get('currentPrice', data['Close'].iloc[-1])
            except:
                company_name = stock_info.get(ticker, ticker)
                current_price = data['Close'].iloc[-1]
            
            # Calculate performance
            month_change = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
            
            # Store data
            portfolio_data[ticker] = {
                'name': company_name,
                'price': current_price,
                'performance': month_change
            }
            
            total_performance += month_change
        
        if not portfolio_data:
            return "I couldn't retrieve data for any of the stocks in your portfolio."
        
        # Calculate average performance
        avg_performance = total_performance / len(portfolio_data)
        
        # Sort stocks by performance
        sorted_stocks = sorted(portfolio_data.items(), key=lambda x: x[1]['performance'], reverse=True)
        
        # Create a table for the portfolio
        response += "<table style='width:100%; border-collapse: collapse;'>"
        
        # Header row
        response += "<tr style='background-color:#f2f2f2;'>"
        response += "<th style='padding:8px; text-align:left;'>Stock</th>"
        response += "<th style='padding:8px; text-align:right;'>Current Price</th>"
        response += "<th style='padding:8px; text-align:right;'>30-Day Return</th>"
        response += "</tr>"
        
        # Add rows for each stock
        for ticker, data in sorted_stocks:
            price = data['price']
            performance = data['performance']
            
            # Determine color based on performance
            color = "green" if performance > 0 else "red"
            
            response += "<tr>"
            response += f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{data['name']} ({ticker})</td>"
            response += f"<td style='padding:8px; text-align:right; border-bottom:1px solid #ddd;'>${price:.2f}</td>"
            response += f"<td style='padding:8px; text-align:right; color:{color}; border-bottom:1px solid #ddd;'>{performance:.2f}%</td>"
            response += "</tr>"
        
        response += "</table>"
        
        # Add portfolio summary
        response += "<br><strong>Portfolio Summary:</strong><br>"
        
        # Overall performance
        color = "green" if avg_performance > 0 else "red"
        response += f"• Average 30-Day Return: <span style='color:{color}'>{avg_performance:.2f}%</span><br>"
        
        # Best and worst performers
        best_stock = sorted_stocks[0]
        worst_stock = sorted_stocks[-1]
        
        response += f"• Best Performer: {best_stock[1]['name']} ({best_stock[0]}) with {best_stock[1]['performance']:.2f}% return<br>"
        response += f"• Worst Performer: {worst_stock[1]['name']} ({worst_stock[0]}) with {worst_stock[1]['performance']:.2f}% return<br>"
        
        # Compare to S&P 500
        try:
            spy = yf.Ticker("SPY")
            spy_data = spy.history(start=start_date, end=end_date)
            spy_performance = ((spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0]) - 1) * 100
            
            if avg_performance > spy_performance:
                outperformance = avg_performance - spy_performance
                response += f"• Your portfolio outperformed the S&P 500 by {outperformance:.2f}% over the past month.<br>"
            else:
                underperformance = spy_performance - avg_performance
                response += f"• Your portfolio underperformed the S&P 500 by {underperformance:.2f}% over the past month.<br>"
        except:
            pass
        
        # Add disclaimer
        response += "<br><em>Disclaimer: Past performance is not indicative of future results. This analysis assumes equal weighting of all stocks.</em>"
        
        return response
    except Exception as e:
        return f"I'm having trouble analyzing your portfolio right now. Please try again later. Error: {str(e)}"

# Main execution
if __name__ == "__main__":
    print("DEBUG: Starting predictor.py", file=sys.stderr)
    print(f"DEBUG: Arguments: {sys.argv}", file=sys.stderr)
    
    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"DEBUG: Processing query: {query}", file=sys.stderr)
        response = process_query(query)
        print(f"DEBUG: Response: {response[:100]}...", file=sys.stderr)
        print(response)
    else:
        print("No query provided. Please ask a question about stocks.")
