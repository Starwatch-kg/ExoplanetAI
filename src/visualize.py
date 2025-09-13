import matplotlib.pyplot as plt

def plot_lightcurve(data, time_col="time", flux_col="flux"):
    """Строит график световой кривой"""
    plt.figure(figsize=(10,5))
    plt.plot(data[time_col], data[flux_col], marker='.', linestyle='-', color='blue')
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.title("Light Curve")
    plt.grid(True)
    plt.show()
