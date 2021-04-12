# EventTool
 
Lightweight tool for analysis and annotation of the events written in juypter-notebook.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [TBD](#tbd)

## General info
Purpose of this tool is to quickly inspect and understand a stream of camera events.
It is written in a form of a simple notebook so that anyone can easily adapt the code for its own needs.
By default, events should be stored in csv file which is then opened and handled by pandas dataframe. 
An example of toy data can be found in data/stream_01.csv and it is used in notebook for demonstration.

## Technologies
* jupyter-notebook
* pandas
* bokeh

## Setup

```
$ pip install -r requirements.txt
$ jupyter-notebook analysis.ipynb
```

## TBD 
* make tutorial how to use it
