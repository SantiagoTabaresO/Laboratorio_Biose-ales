#!/usr/bin/env python
# coding: utf-8

# # Punto 4

# In[12]:


import numpy as np
import matplotlib.pyplot as plt


Fs = 100
k = 2 * (1 + 1) # cedula 1004339631 a = 1
t = np.arange(0, 10, 1/Fs)  

signalxt = 4 * np.cos(80 * np.pi * t + np.pi / 4) + k * np.sin(40 * np.pi * t) + 5

plt.figure(figsize=(20, 6))
plt.plot(t, signalxt, label='x(t)')
plt.title('Señal x(t)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.show()
print(signalxt)



# # punto 5 - 6

# In[18]:


def welch_periodograma(señal, frecuencia_muestreo, tamaño_segmento, solapamiento, ventana='hanning'):
    # Conversión de solapamiento a número de muestras
    tamaño_salto = int(tamaño_segmento * (1 - solapamiento / 100))

    # Calcular el número de segmentos
    num_segmentos = (len(señal) - tamaño_segmento) // tamaño_salto + 1

    # Crear ventana
    if ventana == 'hanning':
        ventana_valores = np.hanning(tamaño_segmento)
    elif ventana == 'blackman':
        ventana_valores = np.blackman(tamaño_segmento)
    else:
        raise ValueError("Tipo de ventana no soportado. Usa 'hanning' o 'blackman'.")

    # Normalización de la ventana
    suma_ventana = np.sum(ventana_valores**2)

    # Inicializar el acumulador para la PSD
    densidad_espectral = np.zeros(tamaño_segmento // 2 + 1)

    # Procesar cada segmento
    for indice in range(num_segmentos):
        # Seleccionar el segmento actual
        inicio = indice * tamaño_salto
        fin = inicio + tamaño_segmento
        segmento_actual = señal[inicio:fin]

        # Verificar que el segmento tenga el tamaño correcto
        if len(segmento_actual) < tamaño_segmento:
            break

        # Aplicar la ventana al segmento
        segmento_ventaneado = segmento_actual * ventana_valores

        # Calcular la DFT del segmento
        fft_segmento = np.fft.rfft(segmento_ventaneado)

        # Calcular el periodograma modificado
        periodograma_modificado = (1 / suma_ventana) * (np.abs(fft_segmento)**2)

        # Acumular el periodograma
        densidad_espectral += periodograma_modificado

    # Promediar los periodogramas
    densidad_espectral /= num_segmentos

    # Calcular las frecuencias
    frecuencias = np.fft.rfftfreq(tamaño_segmento, d=1 / frecuencia_muestreo)

    return frecuencias, densidad_espectral


# Parámetros de Welch
tamaño_segmento = 40  # Tamaño del segmento
solapamiento = 15  # Solapamiento en %
ventana = 'blackman'  # Usando ventana Blackman

# Calcular el periodograma
frecuencias, densidad_espectral = welch_periodograma(xt, Fs, tamaño_segmento, solapamiento, ventana)

# Graficar el resultado
plt.figure(figsize=(10, 6))
plt.plot(frecuencias, 10 * np.log10(densidad_espectral))  # PSD en dB
plt.title('Periodograma de Welch con Ventana Blackman')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad espectral de potencia (PSD) [dB/Hz]')
plt.grid()



