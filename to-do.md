# TODO

1. Remove the chunk of missing data at the beginning, or select only the time period that has data. DONE
2. Relevant features: wind speed, wind direction (from a weather forecast) and past values. We know that the data is seasonal, i.e. it is windier during the winter.
3. Downsample the data from intervals of 10 minutes to 1 hour. DONE
4. Check for irregularity. DONE
5. Either impute missing values with the midpoint, or drop the missing values and accept that the time series will be irregular.
6. Power is negative for low wind speeds. We can’t find any good explanation for why the power would be negative in the literature, and so we should replace negative values with 0. This will also make it easier for certain regression models to predict accurately. DONE
7. Train SARIMA, polynomial regression and neural network models. Polynomial reg: DONE.
8. Make predictions and compare the performance. PARTIAL
9. Threshold values: at 10 m/s wind speed, the pitch of the blades is adjusted so that power remains constant and the turbine doesn’t need to be shutdown.
10. We can mention failure analysis in our report, but we are not going to carry this out. We can  see the failure points in the graph.
