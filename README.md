# BEngThesis

## Application for Analysis of the Economic Growth Indexes for European Countries

*Authors: Agata Makarewicz, Jacek Wiśniewski*

*Faculty of Mathematics & Information Science, Warsaw University of Technology*

### Aim of the work

The aim of the diploma thesis is to apply various clustering algorithms to the time series of economic growth of European countries. The analyzed data will be pre-processed using methods such as segmentation, normalization or anomaly removal. The results of the work will be presented in the form of an application with a graphical user interface written in Django, which will allow the user to compare the indicators for different countries.

### Subject of submitted work

For over 40 years, scientific works have presented various divisions of European countries into economic and cultural groups, based on different criteria such as GDP per capita, the level of industrialization or HDI. Depending on the considered indicators and the date of the analysis, usually, 2 to 5 groups are defined. For instance, in the article written by C. Gräbner et al. (2019), the central, peripheral and Eastern European countries as well as financial centres were distinguished. Students will apply several standard clustering methods such as k-means, hierarchical clustering and the fuzzy c-means method to time series of economic growth to group countries and verify the previously proposed divisions. The algorithms will be evaluated using the existing cluster analysis assessment indexes, e.g. inertia, silhouette score, GAP statistic and PBM index. The thesis will be based on publicly available data, including the Penn World Table. The selection of variables itself is one of the students' tasks. The analysis will cover complete time series and selected segments (e.g. before and after 2008 - the year of the last financial crisis). Students will also pay attention to the aspect of similarity of time series in the context of the assessment of synchronization or non-synchronization of business cycles of selected groups of countries before and after the crisis. The implemented models will be part of the web application in which the user will be able to compare the results of the methods used, select variables and parameters for the models, as well as the development indicators presented in the charts. Visualizations of the clusters obtained with different clustering methods will also be available.

### How to run application

* If you are using application for the first time, run install_requirements.sh file passing python path and R path: install_requirements.sh "{python_path}" "{R_path}"
* Run start_app.sh passing python path: start_app.sh "{python_path}"
