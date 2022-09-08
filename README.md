# trading-bot
Bot de trading que analiza un mercado (forex, materias primas, el que sea) y detecta opciones
de compra siguiendo una estrategia (en construcción).

En el programa "algo_def.py" se introduce un mercado y una temporalidad; se descarga la información
de las acciones del mercado elegido y esta se guarda en un dataframe de pandas. De este
dataframe selecciono una sección. Por ejemplo, si elijo las acciones de Apple como mercado
y la temporalidad diaria, y elijo la sección dataframe[1000:2000] tengo un dataframe desde
el día 1000 al día 2000 desde que hay datos de las acciones de Apple. Con este fragmento puedo
calcular indicadores, encontrar zonas de soporte y resistencia, obtener las velas japonesas
correspondientes, puedo buscar máximos y mínimos y finalmente buscar tendencias ascendentes
o descendentes mediante ajustes de mínimos cuadrados. En la parte del final del programa se
pueden ver las rectas de estos ajustes sobre la gráfica. Todas los cálculos se hacen llamando
a diferentes funciones en el programa "funcions.py", las cuales he creado yo por mi cuenta.

El siguiente paso es programar una estrategia de inversión para este mercado. Una vez encontrada
una estrategia razonable, puedo ponerla a prueba en diferentes pedazos del mercado y calcular la
probabilidad de éxito de una operación. Finalmente, si la estrategia funciona, puedo subir el programa
a la nube (servicios AWS, Google Cloud, etc.) y que haga el análisis del mercado cada cierto tiempo
buscando oportunidades de compra y me las notifique por correo.
