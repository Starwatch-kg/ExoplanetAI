import sys
sys.path.append('./src')
from preprocess import load_lightcurve, normalize
from visualize import plot_lightcurve

if __name__ == "__main__":
    data = load_lightcurve('data/kepler_data.csv')
    print('Данные загружены')
    
    data = normalize(data)
    print('Данные нормализованы')
    
    print('plot_lightcurve function path:', plot_lightcurve.__module__)
    plot_lightcurve(data)
    print(data.head())
