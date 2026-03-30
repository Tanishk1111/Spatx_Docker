#!/usr/bin/env python3
"""Initialize SpatX database and create admin superuser."""
import sys, os
sys.path.insert(0, "/app")

from database import SessionLocal, Base, engine, User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def main():
    print("[INIT] Creating database tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == "admin").first()
        if existing:
            print(f"[OK] Admin user already exists (id={existing.id})")
            return

        admin = User(
            username="admin",
            email="admin@spatx.lab",
            hashed_password=pwd_context.hash("admin123"),
            credits=10000,
            is_active=True
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)
        print(f"[OK] Admin superuser created  —  username: admin  password: admin123  credits: {admin.credits}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
