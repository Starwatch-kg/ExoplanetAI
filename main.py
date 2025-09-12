import sys
sys.path.append('./src')

from preprocess import load_lightcurve, normalize
import visualizelightcurve

if __name__ == "__main__":
    data = load_lightcurve('data/kepler_data.csv')
    print('Данные загружены')
    
    data = normalize(data)
    print('Данные нормализованы')
    
    print('visualizelightcurve module path:', visualizelightcurve.__file__)
    visualizelightcurve.plot_lightcurve(data)
    print(data.head())
