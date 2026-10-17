"""Bootstrap an administrator: PYTHONPATH=backend python backend/manage.py USERNAME."""
import getpass
import sys
from uuid import uuid4
from sqlalchemy import select
from app.services.database import initialize, engine, users
from app.api.auth import hash_password

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python backend/manage.py USERNAME')
    password = getpass.getpass('Administrator password (at least 12 characters): ')
    if len(password) < 12:
        raise SystemExit('Password too short')
    initialize()
    with engine.begin() as connection:
        if connection.execute(select(users.c.id).where(users.c.username == sys.argv[1])).first():
            raise SystemExit('Username already exists')
        connection.execute(users.insert().values(id=str(uuid4()), username=sys.argv[1],
            password_hash=hash_password(password), role='admin', department='IT'))
    print('Administrator created.')
