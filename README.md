# Gemelos digitales de mantenimiento (TPM)

Gemelo digital consultivo para el CNC plasma y el torno. Detecta la degradación del equipo a partir de sus lecturas
(vibración, temperatura, corriente y error de posición o RPM), propone una intervención a tiempo y deja que el jefe de
Mantenimiento la apruebe o la rechace. El sistema **no envía comandos a la máquina**.

Puede trabajar con **señales simuladas** (para evaluar el gemelo) o con **lecturas reales** cargadas desde un CSV.

## Instalación

Requiere Python 3.11 o superior.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Con Docker (el historial, los usuarios y los umbrales se guardan en el volumen `/data`):

```bash
docker build -t gemelo-digital .
docker run -p 8501:8501 -v gemelo_data:/data --env-file .env gemelo-digital
```

Sin Docker, las variables de `.env.example` se definen en el entorno antes de ejecutar (`export GEMELO_MAIL_TO=...`); `.env` solo lo lee Docker con `--env-file`.

Abre http://localhost:8501. Para uso en red, publícalo detrás de un proxy con HTTPS (por ejemplo Nginx o Caddy).

## Primer uso

1. **Usuarios.** Sin archivo `users.json` la aplicación corre en *modo demostración*, sin inicio de sesión y con todos los
   permisos. Para producción crea usuarios con `python crear_usuario.py`.

   | Rol | Puede |
   | --- | --- |
   | `operador` | Consultar todas las vistas |
   | `jefe_mantenimiento` | Consultar y aprobar o rechazar órdenes de trabajo |
   | `admin` | Todo lo anterior y la vista Configuración (umbrales, notificaciones) |

2. **Umbrales.** Los valores por defecto son demostrativos. Ajústalos en la vista **Configuración** con la línea base real
   del equipo y las especificaciones del fabricante. Se guardan en `gemelo_config.json` (ver `gemelo_config.example.json`).
3. **Notificaciones (opcional).** Define las variables de `.env.example` para recibir un correo o un mensaje por webhook
   cuando el gemelo propone una intervención con datos reales. Se avisa una sola vez por intervención.

## Datos reales (CSV)

En la barra lateral elige **Datos reales (CSV)**, selecciona el equipo y sube el archivo. Descarga allí mismo una plantilla.

- Una columna de tiempo (`timestamp`, `fecha` u hora en números) y una columna por variable del equipo, con el mismo nombre
  que en la plantilla. Admite separador `,` o `;`, coma decimal y fechas `2026-01-31 08:05` o `31/01/2026 08:05`.
- El gemelo las lleva a una lectura cada 5 minutos. Los vacíos cuentan como datos perdidos.
- **Las primeras 4 horas deben ser de operación normal**: con ellas el gemelo aprende el perfil normal del equipo.
  Se necesitan al menos 5 horas de datos y como máximo unos 69 días por archivo.
- La intervención se propone cuando el índice de condición supera el umbral durante 4 lecturas seguidas.

Para conectar un gateway MQTT u OPC UA, vuelca sus lecturas a este formato o reemplaza la función `parse_csv` por un lector
del gateway: el análisis posterior no cambia.

## Vistas

| Vista | Para qué sirve |
| --- | --- |
| Telemetría | Señales del equipo con sus umbrales |
| Réplica virtual | Modelo 3D con el estado de cada sensor |
| Alertas | Eventos, orden de trabajo propuesta y decisión del jefe de Mantenimiento |
| Historial | Decisiones y eventos guardados (con descarga en CSV) |
| Indicadores | Disponibilidad, MTBF y MTTR frente a las metas |
| Análisis | Sensibilidad de los supuestos y matriz de pruebas |
| Configuración | Umbrales, notificaciones y usuarios (solo administradores) |
| Ayuda | Cómo usar, metodología e implementación |

Los indicadores de mejora (incremento de MTBF y reducción de MTTR) son **supuestos**, no mediciones.

## Operación

- Datos: `GEMELO_DATA_DIR` contiene `gemelo.db` (SQLite con el historial) y `gemelo.log` (registro rotativo). Respáldalos
  copiando esa carpeta.
- Cambios de umbrales: se aplican al guardar. Si editas `gemelo_config.json` a mano, se leen en la siguiente interacción.
- Las contraseñas se guardan con PBKDF2-SHA256. Tras 5 intentos fallidos se bloquea el inicio de sesión de esa pestaña.

## Limitaciones conocidas

- No lee MQTT ni OPC UA directamente; entra por CSV (ver arriba).
- El perfil normal se aprende de las primeras 4 horas del archivo; si ese tramo ya está degradado, la detección se atrasa.
- Los umbrales por defecto no están calibrados con mediciones reales.
- Con datos reales no existe una comparación reactiva frente a predictiva: ambas franjas muestran las mismas lecturas.
