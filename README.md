# GeoStyle

This repo is the codebase for the paper:

[**GeoStyle: GeoStyle: Discovering Fashion Trends and Events**](https://geostyle.cs.cornell.edu)
<br>
[Utkarsh Mall](http://www.cs.cornell.edu/~utkarshm/), [Kevin Matzen](http://www.kmatzen.com/), [Bharath Hariharan](http://home.bharathh.info/), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [Kavita Bala](http://www.cs.cornell.edu/~kb/)
<br>
ICCV 2019

### Abstract
Understanding fashion styles and trends is of great potential interest to retailers and consumers alike. The photos people upload to social media are a historical and public data source of how people dress across the world and at different times. While we now have tools to automatically recognize the clothes and style attributes of what people are wearing in these photographs, we lack the ability to analyze spatial and temporal trends in these attributes or make predictions about the future. In this paper we address this need by providing an automatic framework (see the figure below) that analyzes large corpora of street imagery to (a) discover and forecast long-term trends of various fashion attributes as well as automatically discovered styles, and (b) identify spatio-temporally localized events that affect what people wear. We show that our framework makes long term trend forecasts that are > 20% more accurate than prior art, and identifies hundreds of socially meaningful events that impact fashion across the globe.

![alt text](https://geostyle.cs.cornell.edu/static/images/pipeline.png)


### Content
The code consists of two directories, ```googlenet``` and ```trends_events```.

```googlenet```: Contains code for training GoogLeNet on StreetStyle-27k and pre-trained models.
<br>
```trends_events```: Contains code for trend-fitting and event detection.
