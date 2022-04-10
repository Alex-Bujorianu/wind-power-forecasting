#TODO

1. Remove the chunk of missing data at the beginning, or select only the time period that has data.
2. Relevant features: wind speed, wind direction (from a weather forecast) and past values. We know that the data is seasonal, i.e. it is windier during the winter.
3. Downsample the data from intervals of 10 minutes to 1 hour.
4. Impute missing values, if necessary (upsampling should be enough).
5. Wind power is negative for low wind speeds. Why?
6. Train SARIMA, polynomial regression and neural network models.
7. Make predictions and compare the performance.
8. Threshold values: if wind speed is too high, power drops to zero because the turbine has to be shutdown. At what wind speed does this occur? Do we use a specialist model or just write an if statement?
9. We can mention failure analysis in our report, but we are not going to carry this out. 
