# bicoba.com
# 1. Inicializar y ajustar el modelo
modelo = Prophet(yearly_seasonality=True, daily_seasonality=False)
modelo.fit(df)

# 2. Crear un DataFrame para el futuro (90 días)
futuro = modelo.make_future_dataframe(periods=90)

# 3. Realizar la predicción
prediccion = modelo.predict(futuro)

# Ver las últimas filas de la predicción
print(prediccion[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
