import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import serial
import time
import collections

muestras_totales = 2000
fs = 1000
ventana_visualizacion = 200 


datos_vivo = collections.deque([0] * ventana_visualizacion, maxlen=ventana_visualizacion)
todos_los_datos = []

plt.ion() 
fig_vivo, ax_vivo = plt.subplots()
linea, = ax_vivo.plot(datos_vivo)
ax_vivo.set_ylim(-2, 2) 
ax_vivo.set_title("Captura de Señal en Tiempo Real (Muestreando...)")
ax_vivo.set_xlabel("Muestras recientes")
ax_vivo.set_ylabel("Voltaje (V)")
ax_vivo.grid(True)

print("--- Iniciando captura visual de 2000 muestras ---")

try:
    ser = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
    usa_simulacion = False
except:
    print("Hardware no detectado. Usando simulación para visualización...")
    usa_simulacion = True

for i in range(muestras_totales):
    if not usa_simulacion:
        linea_ser = ser.readline().decode('utf-8').strip()
        if linea_ser:
            val = float(linea_ser)
    else:
        val = np.random.normal(0, 0.3) + 0.15 * np.sin(2 * np.pi * 60 * i / fs)
        time.sleep(0.01)


    todos_los_datos.append(val)
    datos_vivo.append(val)

    if i % 5 == 0:
        linea.set_ydata(datos_vivo)
        fig_vivo.canvas.draw()
        fig_vivo.canvas.flush_events()
        print(f"Capturando muestra {i+1}/{muestras_totales}...")

plt.ioff() 
plt.close(fig_vivo) 
print("--- Captura finalizada. Generando análisis ---")

datos_np = np.array(todos_los_datos)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(datos_np, bins=40, color='#1f77b4', alpha=0.7, edgecolor='black')
plt.title('RESULTADO 1: Histograma (Distribución Gaussiana)')
plt.xlabel('Voltaje')

plt.subplot(2, 2, 2)
stats.probplot(datos_np, dist="norm", plot=plt)
plt.title('RESULTADO 2: Gráfica Q-Q (Prueba de Normalidad)')
plt.gca().get_lines()[0].set_markerfacecolor('#1f77b4')
plt.gca().get_lines()[0].set_markeredgecolor('#1f77b4')
plt.gca().get_lines()[1].set_color('red') 


plt.subplot(2, 1, 2)
fft_vals = np.abs(np.fft.fft(datos_np))
freqs = np.fft.fftfreq(muestras_totales, 1/fs)
mask = freqs >= 0 
plt.plot(freqs[mask], fft_vals[mask], color='#ff7f0e')
plt.title('RESULTADO 3: Espectro de Frecuencia (Detección de Ruido de Red)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid(True, which='both', linestyle='--')
plt.axvspan(55, 65, color='red', alpha=0.2, label='Zona 60Hz')
plt.legend()

plt.tight_layout()
plt.show()
