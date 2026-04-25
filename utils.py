import re

def clean_feature_name(name):
    """
    Converts cryptic sensor feature names into human-readable versions.
    Example: 'tBodyAcc-mean()-X' -> 'Body Acceleration Mean X'
    """
    # Replace prefixes
    name = name.replace('tBody', 'Body ')
    name = name.replace('tGravity', 'Gravity ')
    name = name.replace('fBody', 'Frequency Body ')
    name = name.replace('fGravity', 'Frequency Gravity ')
    
    # Replace signal types
    name = name.replace('AccJerk', ' Acceleration Jerk')
    name = name.replace('AccMag', ' Acceleration Magnitude')
    name = name.replace('Acc', ' Acceleration')
    name = name.replace('GyroJerk', ' Gyroscope Jerk')
    name = name.replace('GyroMag', ' Gyroscope Magnitude')
    name = name.replace('Gyro', ' Gyroscope')
    
    # Replace statistics
    name = name.replace('-mean()', ' Mean')
    name = name.replace('-std()', ' Std Dev')
    name = name.replace('-mad()', ' Median Abs Dev')
    name = name.replace('-max()', ' Max')
    name = name.replace('-min()', ' Min')
    name = name.replace('-sma()', ' Signal Magnitude Area')
    name = name.replace('-energy()', ' Energy')
    name = name.replace('-iqr()', ' Interquartile Range')
    name = name.replace('-entropy()', ' Entropy')
    name = name.replace('-arCoeff()', ' Autoregression Coeff')
    name = name.replace('-correlation()', ' Correlation')
    name = name.replace('-maxInds', ' Max Index')
    name = name.replace('-meanFreq()', ' Mean Frequency')
    name = name.replace('-skewness()', ' Skewness')
    name = name.replace('-kurtosis()', ' Kurtosis')
    name = name.replace('-bandsEnergy()', ' Bands Energy')
    
    # Replace Axis
    name = name.replace('-X', ' X')
    name = name.replace('-Y', ' Y')
    name = name.replace('-Z', ' Z')
    
    # Clean up multiple spaces and comma/brackets
    name = name.replace(',', ' ')
    name = name.replace('(', '')
    name = name.replace(')', '')
    name = re.sub(' +', ' ', name).strip()
    
    # Capitalize each word properly
    name = ' '.join(word.capitalize() if word.islower() else word for word in name.split())
    
    return name

def get_readable_mapping(feature_names):
    """Returns a dictionary mapping original names to cleaned names."""
    return {name: clean_feature_name(name) for name in feature_names}
