
# NEWS MOOD

The assignment consists in perform a sentiment analysis of the Twitter activity of the following news oulets: BBC, CBS, CNN, Fox, and New York Times, and to present the final output a visualized summary of the sentiments expressed in Tweets.



```python
# Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import tweepy
import yaml
import seaborn as sns

```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Given a yaml filename , return the contents of that file

def get_file_contents(filename):
    try:
        with open(filename, 'r') as config_file:
            config = yaml.load(config_file)
            return (config)
    except FileNotFoundError:
        print("'%s' file not found" % filename)
        
```


```python
TWITTER_CONFIG_FILE = 'auth.yaml'

config = get_file_contents(TWITTER_CONFIG_FILE)

# Twitter API Keys
consumer_key = config['twitter']['consumer_key']
consumer_secret = config['twitter']['consumer_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target Search Term
news_outlets = ("BBC", "CBS", "CNN", "FoxNews", "nytimes")

```

### Extract  tweets for each News Outlet and perform a  Vader sentiment analysis on each tweet.


```python
# Create arrays for holding all the sentiments for each News Outlet

sentiments = []
sentiments_means = []
sentiments_means_df = pd.DataFrame()

positive =0
negative =0
neutral =0

# Loop through all target News Outlets
for outlet in news_outlets:

    counter = 1

    # Variables for holding sentiments
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []

    public_tweets = api.search(outlet, count=100, result_type="recent")

    # Loop through all tweets
    for tweet in public_tweets["statuses"]:

        # Run Vader Analysis on each tweet
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
            
        # Add each value to the appropriate array
        compound_list.append(compound)
        positive_list.append(pos)
        negative_list.append(neg)
        neutral_list.append(neu)
        
        # Add sentiments for each tweet into an array
        sentiments.append({ "Media Source": outlet,
                            "Text" : tweet["text"],
                            "Date": tweet["created_at"], 
                            "Compound": compound,
                            "Positive": pos,
                            "Neutral": neu,
                            "Negative": neg,
                            "Tweets Ago": counter})
        
        if (compound > 0.2) :
            positive += 1
        elif (compound < -0.2):
            negative += 1
        else:
            neutral +=1
                
        # Add to counter 
        counter += 1

    # Store the Average Sentiments
    sentiments_means = { "Media Source": outlet,
                        "Compound": np.mean(compound_list),
                        "Positive": np.mean(positive_list),
                        "Neutral": np.mean(neutral_list),
                        "Negative": np.mean(negative_list),
                        "Tweet Count": len(compound_list)
    }
    
    sentiments_means_df=sentiments_means_df.append(sentiments_means,ignore_index=True)
    
```


```python
# Convert sentiments and sentiments_means_df dictionaries to DataFrames and Export the Data into CSV files
sentiments_df = pd.DataFrame.from_dict(sentiments)

sentiments_means_df[['Compound','Negative','Neutral', 'Positive']] = sentiments_means_df[['Compound',
                                                                                          'Negative',
                                                                                          'Neutral', 
                                                                                          'Positive']].apply(pd.to_numeric)

sentiments_means_df.to_csv("HW_News_Mood.csv", encoding="utf-8", index=False)

# Sort each plot point by its relative timestamp
sentiments_df.sort_values(by="Date", ascending=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Media Source</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>182</th>
      <td>-0.3182</td>
      <td>Mon Jan 08 23:20:25 +0000 2018</td>
      <td>CBS</td>
      <td>0.126</td>
      <td>0.874</td>
      <td>0.000</td>
      <td>RT @WiredSources: Pentagon: The Trump Administ...</td>
      <td>83</td>
    </tr>
    <tr>
      <th>181</th>
      <td>0.3612</td>
      <td>Mon Jan 08 23:20:25 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>0.906</td>
      <td>0.094</td>
      <td>RT @RaniaKhalek: Head of RT to CBS about US in...</td>
      <td>82</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:20:32 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @JackPosobiec: Couldn't you have just used ...</td>
      <td>81</td>
    </tr>
    <tr>
      <th>179</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:20:32 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @dytpqdlxn: 📣)) 광화문 광장에서 한기총·CBS 규탄대회, \n한기...</td>
      <td>80</td>
    </tr>
    <tr>
      <th>178</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:20:36 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @dytpqdlxn: 🔥광화문 신천지, 반국가 반사회 반종교는 거짓말하는 한기...</td>
      <td>79</td>
    </tr>
    <tr>
      <th>176</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:20:42 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>장준환 감독은 7일 CBS노컷뉴스에 "우리로서는 정말 든든했다"면서 "문재인 대통령...</td>
      <td>77</td>
    </tr>
    <tr>
      <th>177</th>
      <td>-0.7717</td>
      <td>Mon Jan 08 23:20:42 +0000 2018</td>
      <td>CBS</td>
      <td>0.271</td>
      <td>0.729</td>
      <td>0.000</td>
      <td>We are going to substantialy reduce taxes and ...</td>
      <td>78</td>
    </tr>
    <tr>
      <th>175</th>
      <td>0.7845</td>
      <td>Mon Jan 08 23:20:50 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>0.635</td>
      <td>0.365</td>
      <td>Steelers To Offer Super Bowl Tickets, Other Pr...</td>
      <td>76</td>
    </tr>
    <tr>
      <th>174</th>
      <td>0.4019</td>
      <td>Mon Jan 08 23:20:59 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>0.876</td>
      <td>0.124</td>
      <td>RT @AynRandPaulRyan: Rose\nHalperin\nLauer\nTh...</td>
      <td>75</td>
    </tr>
    <tr>
      <th>173</th>
      <td>0.6369</td>
      <td>Mon Jan 08 23:20:59 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>0.826</td>
      <td>0.174</td>
      <td>RT @cbssecmtown: CBS senior hurlers take on @h...</td>
      <td>74</td>
    </tr>
    <tr>
      <th>172</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:21:00 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@MateuszGotz na filmie CBŚ z zatrzymania (w TV...</td>
      <td>73</td>
    </tr>
    <tr>
      <th>171</th>
      <td>0.4019</td>
      <td>Mon Jan 08 23:21:06 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>0.876</td>
      <td>0.124</td>
      <td>RT @AynRandPaulRyan: Rose\nHalperin\nLauer\nTh...</td>
      <td>72</td>
    </tr>
    <tr>
      <th>170</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:21:07 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @zooroo616: 기독교 핫이슈! 신천지예수교회 급성장의 이유는? \n\n...</td>
      <td>71</td>
    </tr>
    <tr>
      <th>168</th>
      <td>-0.7506</td>
      <td>Mon Jan 08 23:21:14 +0000 2018</td>
      <td>CBS</td>
      <td>0.234</td>
      <td>0.766</td>
      <td>0.000</td>
      <td>The hate for President Trump expressed on news...</td>
      <td>69</td>
    </tr>
    <tr>
      <th>169</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:21:14 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@cftheking @mari_ness @D_Libris The deal with ...</td>
      <td>70</td>
    </tr>
    <tr>
      <th>167</th>
      <td>-0.0772</td>
      <td>Mon Jan 08 23:21:17 +0000 2018</td>
      <td>CBS</td>
      <td>0.091</td>
      <td>0.909</td>
      <td>0.000</td>
      <td>@TheGreatFeather @On_The_Hook @MyBrianLeyh @MS...</td>
      <td>68</td>
    </tr>
    <tr>
      <th>166</th>
      <td>0.4927</td>
      <td>Mon Jan 08 23:21:19 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>0.824</td>
      <td>0.176</td>
      <td>@marjenjr @WildHeartAshley @YandR_CBS @malyoun...</td>
      <td>67</td>
    </tr>
    <tr>
      <th>165</th>
      <td>-0.5423</td>
      <td>Mon Jan 08 23:21:19 +0000 2018</td>
      <td>CBS</td>
      <td>0.149</td>
      <td>0.851</td>
      <td>0.000</td>
      <td>RT @tlrd: Alan Cumming to Be First Openly Gay ...</td>
      <td>66</td>
    </tr>
    <tr>
      <th>164</th>
      <td>-0.5423</td>
      <td>Mon Jan 08 23:21:23 +0000 2018</td>
      <td>CBS</td>
      <td>0.200</td>
      <td>0.800</td>
      <td>0.000</td>
      <td>RT @PFF: Throwing at these CBs was a bad idea ...</td>
      <td>65</td>
    </tr>
    <tr>
      <th>163</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:21:24 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @zooroo616: 신천지예수교회가 급성장하는 이유는 뭐죠? \n\nhttp...</td>
      <td>64</td>
    </tr>
    <tr>
      <th>162</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:21:27 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @AndraDayMusic: TONIGHT Andra Day and @comm...</td>
      <td>63</td>
    </tr>
    <tr>
      <th>161</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:21:32 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@NorahODonnell @CBSThisMorning Wait. Some jour...</td>
      <td>62</td>
    </tr>
    <tr>
      <th>160</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:21:33 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @swatcbs: #SWAT star @ShemarMoore heats up ...</td>
      <td>61</td>
    </tr>
    <tr>
      <th>159</th>
      <td>-0.2960</td>
      <td>Mon Jan 08 23:21:36 +0000 2018</td>
      <td>CBS</td>
      <td>0.196</td>
      <td>0.804</td>
      <td>0.000</td>
      <td>@CrimMinds_CBS Finally Red is backup, i missed...</td>
      <td>60</td>
    </tr>
    <tr>
      <th>158</th>
      <td>-0.4173</td>
      <td>Mon Jan 08 23:21:45 +0000 2018</td>
      <td>CBS</td>
      <td>0.166</td>
      <td>0.834</td>
      <td>0.000</td>
      <td>@TheGreatFeather @KatTheHammer1 @MyBrianLeyh @...</td>
      <td>59</td>
    </tr>
    <tr>
      <th>157</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:21:46 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@DretsCat @Reuters @AP @BBCNews @guardian @Ind...</td>
      <td>58</td>
    </tr>
    <tr>
      <th>156</th>
      <td>-0.5423</td>
      <td>Mon Jan 08 23:21:59 +0000 2018</td>
      <td>CBS</td>
      <td>0.149</td>
      <td>0.851</td>
      <td>0.000</td>
      <td>RT @tlrd: Alan Cumming to Be First Openly Gay ...</td>
      <td>57</td>
    </tr>
    <tr>
      <th>155</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:22:02 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @SandruSom: @SoapDigest @boldinsider @BandB...</td>
      <td>56</td>
    </tr>
    <tr>
      <th>154</th>
      <td>-0.4466</td>
      <td>Mon Jan 08 23:22:09 +0000 2018</td>
      <td>CBS</td>
      <td>0.423</td>
      <td>0.577</td>
      <td>0.000</td>
      <td>@CrimMinds_CBS I MISSED HIM #CriminalMinds htt...</td>
      <td>55</td>
    </tr>
    <tr>
      <th>153</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:22:10 +0000 2018</td>
      <td>CBS</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @moonsich: 장준환 감독은 7일 CBS노컷뉴스에 "우리로서는 정말 든든...</td>
      <td>54</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>292</th>
      <td>-0.2500</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>FoxNews</td>
      <td>0.083</td>
      <td>0.917</td>
      <td>0.000</td>
      <td>RT @Hoosiers1986: As if we needed proof that t...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>183</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>CNN</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @CNNEE: Así se ven algunos supermercados en...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>184</th>
      <td>-0.5106</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>CNN</td>
      <td>0.283</td>
      <td>0.571</td>
      <td>0.146</td>
      <td>RT @GeorgeTakei: As much as Donald would like ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>185</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>CNN</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @linbea945: Should I sell the story to CNN ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>187</th>
      <td>0.4767</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>CNN</td>
      <td>0.000</td>
      <td>0.853</td>
      <td>0.147</td>
      <td>RT @cnnbrk: Oprah Winfrey is “actively thinkin...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>188</th>
      <td>0.6597</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>CNN</td>
      <td>0.000</td>
      <td>0.795</td>
      <td>0.205</td>
      <td>RT @CNN: JUST IN: Oprah Winfrey is actively th...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>189</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>CNN</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @1ClickBiz: Newt Gingrich: CNN's Jake Tappe...</td>
      <td>7</td>
    </tr>
    <tr>
      <th>293</th>
      <td>0.8979</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>FoxNews</td>
      <td>0.000</td>
      <td>0.562</td>
      <td>0.438</td>
      <td>RT @FoxNews: .@HerschelWalker: "Donald Trump i...</td>
      <td>11</td>
    </tr>
    <tr>
      <th>294</th>
      <td>-0.3182</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>FoxNews</td>
      <td>0.126</td>
      <td>0.874</td>
      <td>0.000</td>
      <td>🥊✊️🗣@KevinJacksonTBS: "[Barack Obama] lost 17,...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>395</th>
      <td>0.4767</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>0.871</td>
      <td>0.129</td>
      <td>RT @SangerNYT: For several months @WilliamJBro...</td>
      <td>13</td>
    </tr>
    <tr>
      <th>396</th>
      <td>0.3818</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>0.894</td>
      <td>0.106</td>
      <td>RT @RBReich: We must not normalize the fact th...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>186</th>
      <td>0.2960</td>
      <td>Mon Jan 08 23:25:23 +0000 2018</td>
      <td>CNN</td>
      <td>0.000</td>
      <td>0.804</td>
      <td>0.196</td>
      <td>RT @YourTumblrFeed: South Korea: don't tell an...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>288</th>
      <td>-0.4939</td>
      <td>Mon Jan 08 23:25:24 +0000 2018</td>
      <td>FoxNews</td>
      <td>0.160</td>
      <td>0.763</td>
      <td>0.076</td>
      <td>RT @marklevinshow: There is this: Top FBI Russ...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>287</th>
      <td>-0.6669</td>
      <td>Mon Jan 08 23:25:24 +0000 2018</td>
      <td>FoxNews</td>
      <td>0.223</td>
      <td>0.777</td>
      <td>0.000</td>
      <td>@FoxNews I think he needs to concentrate less ...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>394</th>
      <td>0.4810</td>
      <td>Mon Jan 08 23:25:24 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>0.876</td>
      <td>0.124</td>
      <td>RT @JoyAnnReid: Tucked into that Friday @NYTim...</td>
      <td>12</td>
    </tr>
    <tr>
      <th>389</th>
      <td>0.4810</td>
      <td>Mon Jan 08 23:25:24 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>0.876</td>
      <td>0.124</td>
      <td>RT @JoyAnnReid: Tucked into that Friday @NYTim...</td>
      <td>7</td>
    </tr>
    <tr>
      <th>390</th>
      <td>0.8176</td>
      <td>Mon Jan 08 23:25:24 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>0.717</td>
      <td>0.283</td>
      <td>I love how people who are not neuroscientists ...</td>
      <td>8</td>
    </tr>
    <tr>
      <th>391</th>
      <td>-0.2960</td>
      <td>Mon Jan 08 23:25:24 +0000 2018</td>
      <td>nytimes</td>
      <td>0.135</td>
      <td>0.779</td>
      <td>0.086</td>
      <td>RT @nytimes: Energy Secretary Rick Perry’s pla...</td>
      <td>9</td>
    </tr>
    <tr>
      <th>392</th>
      <td>0.4810</td>
      <td>Mon Jan 08 23:25:24 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>0.876</td>
      <td>0.124</td>
      <td>RT @JoyAnnReid: Tucked into that Friday @NYTim...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>393</th>
      <td>0.6880</td>
      <td>Mon Jan 08 23:25:24 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>0.831</td>
      <td>0.169</td>
      <td>RT @amyinthelou: @nytimes This would be a real...</td>
      <td>11</td>
    </tr>
    <tr>
      <th>283</th>
      <td>0.4019</td>
      <td>Mon Jan 08 23:25:25 +0000 2018</td>
      <td>FoxNews</td>
      <td>0.000</td>
      <td>0.803</td>
      <td>0.197</td>
      <td>RT @foxnewspolitics: Trump administration ends...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>284</th>
      <td>0.6580</td>
      <td>Mon Jan 08 23:25:25 +0000 2018</td>
      <td>FoxNews</td>
      <td>0.000</td>
      <td>0.732</td>
      <td>0.268</td>
      <td>@FoxNews @POTUS Wow I actually think I can see...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>285</th>
      <td>0.4588</td>
      <td>Mon Jan 08 23:25:25 +0000 2018</td>
      <td>FoxNews</td>
      <td>0.000</td>
      <td>0.812</td>
      <td>0.188</td>
      <td>RT @FoxNews: Grand jury seeks testimony about ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>286</th>
      <td>-0.4404</td>
      <td>Mon Jan 08 23:25:25 +0000 2018</td>
      <td>FoxNews</td>
      <td>0.121</td>
      <td>0.879</td>
      <td>0.000</td>
      <td>RT @FoxNews: .@kimguilfoyle on attacks on @POT...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>384</th>
      <td>0.3818</td>
      <td>Mon Jan 08 23:25:25 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>0.894</td>
      <td>0.106</td>
      <td>RT @RBReich: We must not normalize the fact th...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>385</th>
      <td>-0.5556</td>
      <td>Mon Jan 08 23:25:25 +0000 2018</td>
      <td>nytimes</td>
      <td>0.187</td>
      <td>0.813</td>
      <td>0.000</td>
      <td>This is why I could never join a fraternity. I...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>386</th>
      <td>0.0000</td>
      <td>Mon Jan 08 23:25:25 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Everybody has their own taste! https://t.co/32...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>388</th>
      <td>-0.6124</td>
      <td>Mon Jan 08 23:25:25 +0000 2018</td>
      <td>nytimes</td>
      <td>0.174</td>
      <td>0.826</td>
      <td>0.000</td>
      <td>RT @DavidLeopold: If you think Trump’s Stephen...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>387</th>
      <td>0.3818</td>
      <td>Mon Jan 08 23:25:25 +0000 2018</td>
      <td>nytimes</td>
      <td>0.000</td>
      <td>0.890</td>
      <td>0.110</td>
      <td>RT @Imperator_Rex3: @nytimes @NYT is soon to b...</td>
      <td>5</td>
    </tr>
    <tr>
      <th>383</th>
      <td>0.6908</td>
      <td>Mon Jan 08 23:25:26 +0000 2018</td>
      <td>nytimes</td>
      <td>0.087</td>
      <td>0.646</td>
      <td>0.266</td>
      <td>RT @AliceRothchild: Boycott, divestment, and s...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>483 rows × 8 columns</p>
</div>




```python
print ("Positive tweets : {:3d} , Negative tweets: {:3d}, and Neutral tweets: {:3d}".format(positive, negative, neutral))

```

    Positive tweets : 151 , Negative tweets: 112, and Neutral tweets: 220
    

### Scatter Plot of Sentiments of the last 100 tweets sent out by each News Outlet


```python
#sns.lmplot(x="Tweets Ago",y="Compound", data=sentiments_df, hue="Media Source", scatter= True, 
#           palette= "Set1", markers= "o", fit_reg=False, size=7, aspect=.8, legend_out= True)

g = sns.FacetGrid(sentiments_df, hue="Media Source", size=6, legend_out= True)
g = (g.map(plt.scatter, "Tweets Ago", "Compound", edgecolor="black", linewidth=.5).add_legend())

#Incorporate the other graph properties
plt.grid(True, ls='dashed')
plt.axis([102, 0, -1, 1])
plt.title("Sentiment Analysis of Media Tweets (%s)" % time.strftime(" %x"), fontsize=15)
plt.xlabel("Tweets Ago", fontsize=14)
plt.ylabel("Tweet Polarity", fontsize=14)
plt.savefig('charts/Sent_Analysis_News_fig1.png')
plt.show()

```


![png](HW_News-Mood_files/HW_News-Mood_10_0.png)


### CONCLUSIONS: 

The trends were all over the place. There is not identiable trend. The data are, on average, neutral.
The sentiment was more negative two days ago. 

### Bar Plot visualizing the overall sentiments of the last 100 tweets from each news outlet.


```python

sns.factorplot(x= "Media Source",y="Compound", data=sentiments_means_df, kind="bar", size=7, aspect=.8, palette= "Set1")

#Incorporate the other graph properties

plt.title("Overall Media Sentiment based on Twitter (%s)" % time.strftime(" %x"), fontsize=14)
plt.ylabel("Tweet Polarity", fontsize=14)
plt.xlabel("Media Source", fontsize=14)
plt.grid(True, ls='dashed')
plt.hlines(0, -1, 10, colors='k')
plt.savefig('charts/Overall_Media_Sent_fig2.png')
plt.show()
```


![png](HW_News-Mood_files/HW_News-Mood_13_0.png)


### CONCLUSIONS: 

In this graph, FoxNews has more positive tweets on January 8th of 2018 afternoon, maybe because many
people tweet about Golden Globe Awards. 

In addition, it would be better to analyze in a longer period of time because this analysis of only 100 tweets per organization or Media Outlet depends on at what time is done.
