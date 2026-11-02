"""Crea o actualiza un usuario en users.json (la contraseña se guarda con hash, nunca en texto plano).

Uso:  python crear_usuario.py [ruta/users.json]
Roles: operador (solo consulta) · jefe_mantenimiento (consulta y decide órdenes) · admin (todo, incluida la configuración)
"""
import getpass
import hashlib
import json
import os
import sys
from pathlib import Path

ITERATIONS = 200_000
ROLES = ("operador", "jefe_mantenimiento", "admin")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEMELO_USERS", "users.json"))
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"users": {}}
    username = input("Usuario: ").strip()
    name = input("Nombre completo: ").strip() or username
    role = input(f"Rol ({'/'.join(ROLES)}): ").strip()
    if not username or role not in ROLES:
        sys.exit("Usuario vacío o rol inválido.")
    password = getpass.getpass("Contraseña (mínimo 8 caracteres): ")
    if len(password) < 8 or password != getpass.getpass("Repite la contraseña: "):
        sys.exit("La contraseña es muy corta o no coincide.")
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    data.setdefault("users", {})[username] = {
        "name": name, "role": role, "salt": salt.hex(), "hash": digest.hex(), "iterations": ITERATIONS,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Usuario '{username}' guardado en {path}")


if __name__ == "__main__":
    main()
