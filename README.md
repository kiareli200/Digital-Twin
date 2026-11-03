# Gemelo digital de mantenimiento · CNC plasma y torno

Aplicación Streamlit que recibe las lecturas del gateway de planta por **MQTT**, aprende el comportamiento normal de cada equipo, advierte cuando se aleja de él y propone una orden de trabajo que aprueba el jefe de Mantenimiento. También tiene un **modo simulación** para evaluar el gemelo antes de instalar sensores.

## Cómo llegan los datos

```
Sensores → DAQ / ESP32 (gateway) → broker MQTT de la planta → gemelo (suscriptor) → base SQLite → análisis
```

El gemelo se suscribe al broker; no se sube ningún archivo.

| Dato | Topic | Mensaje |
|---|---|---|
| Lectura | `planta/<equipo>/<variable>` | `{"ts": 1767225600, "value": 2.4}` o solo el número |
| Lectura con varios puntos | `planta/<equipo>/<variable>/<punto>` | igual (el gemelo usa el peor punto) |
| Alarma del controlador | `planta/<equipo>/alarma` | `{"code": "E12", "message": "Sobrecorriente eje Y"}` |

- Equipos: `cnc-01`, `tor-01`. Variables: `vibracion`, `temperatura`, `corriente`, `rpm` (torno).
- `ts` puede ser epoch en segundos, milisegundos o fecha ISO; si falta, se usa la hora de llegada. Fechas en el futuro se corrigen.
- Mensajes con topic o valor inválido se descartan y se cuentan en el panel de Configuración.
- Las lecturas se promedian cada 5 minutos y se conservan `GEMELO_RETENTION_DAYS` días.

## Cómo decide

1. **Aprendizaje**: las primeras 4 horas de lecturas tras el inicio (o tras registrar un mantenimiento ejecutado) definen el perfil normal del equipo.
2. **Detección**: una variable debe quedar fuera de lo normal en 4 lecturas consecutivas para generar advertencia. Así un pico aislado no dispara nada.
3. **Propuesta**: si el índice de condición supera el umbral, se propone intervenir. El jefe aprueba o rechaza y queda registrado con fecha y usuario.
4. **Aviso**: correo y/o webhook, una sola vez por equipo dentro de `GEMELO_ALERT_COOLDOWN_H`.
5. **Mantenimiento ejecutado**: el botón en Alertas reinicia el aprendizaje para el equipo reparado.

## Indicadores medidos

En **Historial** se registran las paradas (equipo, inicio, fin, nota). En **Indicadores** se calcula, para el periodo elegido, MTBF, MTTR y disponibilidad medidos y se comparan con la línea base (CNC: MTBF 32,44 h · MTTR 5,06 h · disponibilidad 86,5 %) y las metas.

## Instalación

```bash
pip install -r requirements.txt
cp .env.example .env          # completar GEMELO_MQTT_HOST y lo demás que se use
streamlit run app.py
```

Con Docker: `docker build -t gemelo . && docker run -p 8501:8501 --env-file .env -v gemelo-data:/data gemelo`.

El servidor del gemelo debe poder alcanzar el broker de la planta (red segmentada, usuario/contraseña y TLS con `GEMELO_MQTT_TLS=1`).

### Probar sin sensores

```bash
python simulador_gateway.py --host <broker> --equipo CNC-01 --escenario desgaste --historia 48
```

Publica 48 h de historia con degradación y sigue en tiempo real. Escenarios: `normal`, `desgaste`, `falla`.

### Usuarios y roles

Sin `users.json` la app se abre sin inicio de sesión. Para producción: `python crear_usuario.py` (roles `operador`, `jefe_mantenimiento`, `admin`).

## Alcance y límites actuales

- La entrada es **solo MQTT**. OPC UA no está implementado; si el DAQ lo usa, un conversor OPC UA→MQTT en el gateway resuelve la conexión.
- Las señales del controlador (F2300BT) entran como alarmas en `planta/<equipo>/alarma`; requieren el conversor de contingencia.
- El error de posición del CNC se captura por inspección TPM, no por sensor.
- Los umbrales son demostrativos: deben calibrarse con la línea base real (Configuración → Umbrales).
- El gemelo es consultivo: propone, no actúa sobre las máquinas.

## Pruebas

```bash
pip install pytest amqtt && python -m pytest tests
```
