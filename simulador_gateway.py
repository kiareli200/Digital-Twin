"""Simulador del gateway de planta: publica por MQTT lo mismo que enviaría el ESP32/DAQ real.

Sirve para probar el gemelo sin tener los sensores instalados. Requiere: pip install paho-mqtt

Ejemplos
  python simulador_gateway.py --host localhost --equipo CNC-01 --escenario desgaste --historia 48
  python simulador_gateway.py --host broker.planta.local --user gemelo --password ... --equipo TOR-01 --velocidad 60

Publica en  planta/<equipo>/<variable>  con payload {"ts": <epoch>, "value": <numero>}
  variables: vibracion, temperatura, corriente, rpm (solo torno)
"""
import argparse
import json
import math
import random
import time

import paho.mqtt.client as mqtt

# (base, ruido, deriva al final del desgaste)
EQUIPOS = {
    "CNC-01": {"vibracion": (1.35, 0.10, 4.2), "temperatura": (43.0, 0.65, 39.0), "corriente": (8.8, 0.22, 10.5)},
    "TOR-01": {"vibracion": (1.55, 0.12, 4.6), "temperatura": (45.0, 0.70, 37.0), "corriente": (12.2, 0.30, 13.0), "rpm": (900.0, 5.5, -165.0)},
}
ESCENARIOS = {"normal": None, "desgaste": 0.6, "falla": 0.25}  # fracción de la corrida en la que empieza a degradarse


def valor(equipo, variable, progreso, escenario):
    base, ruido, deriva = EQUIPOS[equipo][variable]
    inicio = ESCENARIOS[escenario]
    d = 0.0
    if inicio is not None and progreso > inicio:
        d = deriva * ((progreso - inicio) / (1 - inicio)) ** 2
    ciclo = 1 + 0.02 * math.sin(progreso * 60)
    return round(base * ciclo + d + random.gauss(0, ruido), 3)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--host", default="localhost")
    a.add_argument("--port", type=int, default=1883)
    a.add_argument("--user")
    a.add_argument("--password")
    a.add_argument("--prefijo", default="planta")
    a.add_argument("--equipo", choices=EQUIPOS, default="CNC-01")
    a.add_argument("--escenario", choices=ESCENARIOS, default="desgaste")
    a.add_argument("--historia", type=float, default=48, help="horas de historia que se envían de golpe al iniciar")
    a.add_argument("--duracion", type=float, default=72, help="horas totales del escenario (la historia es el tramo ya recorrido)")
    a.add_argument("--intervalo", type=float, default=60, help="segundos entre lecturas (tiempo simulado)")
    a.add_argument("--velocidad", type=float, default=1, help="1 = tiempo real; 60 = una hora simulada por minuto")
    a.add_argument("--tls", action="store_true")
    args = a.parse_args()

    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{args.equipo}")
    if args.user:
        cli.username_pw_set(args.user, args.password)
    if args.tls:
        cli.tls_set()
    cli.connect(args.host, args.port)
    cli.loop_start()

    paso = args.intervalo
    total = args.duracion * 3600
    ahora = time.time()
    t_sim = -args.historia * 3600  # segundos respecto de ahora; las lecturas históricas llevan su fecha real
    enviados = 0
    print(f"Enviando {args.equipo} · escenario {args.escenario} · {args.historia:g} h de historia")
    try:
        while t_sim <= total - args.historia * 3600:
            progreso = (t_sim + args.historia * 3600) / total
            ts = ahora + t_sim if t_sim <= 0 else time.time()
            for var in EQUIPOS[args.equipo]:
                cli.publish(f"{args.prefijo}/{args.equipo.lower()}/{var}",
                            json.dumps({"ts": round(ts, 3), "value": valor(args.equipo, var, progreso, args.escenario)}), qos=1)
                enviados += 1
            t_sim += paso
            if t_sim > 0:
                time.sleep(paso / args.velocidad)
            elif enviados % 500 == 0:
                time.sleep(0.05)  # no saturar el broker con el historial
        print(f"Listo: {enviados} lecturas enviadas")
    except KeyboardInterrupt:
        print(f"Detenido: {enviados} lecturas enviadas")
    finally:
        cli.loop_stop()
        cli.disconnect()


if __name__ == "__main__":
    main()
