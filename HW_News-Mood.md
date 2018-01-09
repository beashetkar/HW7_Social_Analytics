
# NEWS MOOD

The assignment consists in perform a sentiment analysis of the Twitter activity of the following news oulets: BBC, CBS, CNN, Fox, and New York Times, and to present the final output a visualized summary of the sentiments expressed in Tweets.


### Needs the following programs and libraries installed:

tweepy==3.5.0
vaderSentiment==2.5
seaborn==0.8
PyYAML==3.12
python-dateutil==2.6.1
pandas==0.20.3
matplotlib==2.1.0
numpy==1.13.3
jupyter==1.0.0
jupyter-client==5.1.0
jupyter-console==5.2.0
jupyter-core==4.3.0

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


![png](HW_News-Mood_files/HW_News-Mood_10_0.png)


### CONCLUSIONS: 

The trends were all over the place. There is not identiable trend. The data are, on average, neutral.
The sentiment was more negative two days ago. 

### Bar Plot visualizing the overall sentiments of the last 100 tweets from each news outlet.


![png](HW_News-Mood_files/HW_News-Mood_13_0.png)


### CONCLUSIONS: 

In this graph, FoxNews has more positive tweets on January 8th of 2018 afternoon, maybe because many
people tweet about Golden Globe Awards. 

In addition, it would be better to analyze in a longer period of time because this analysis of only 100 tweets per organization or Media Outlet depends on at what time is done.
