#!/usr/bin/env python3
"""
Initialize SpatX Database
Creates the SQLite database and adds a default admin user
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal, Base, engine, User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def create_admin_user(username="admin", email="admin@spatx.lab", password="admin123"):
    """Create default admin user"""
    db = SessionLocal()
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            print(f"⚠️  User '{username}' already exists")
            return existing_user
        
        # Create new admin user with 1000 credits
        hashed_password = pwd_context.hash(password)
        admin_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            credits=1000,
            is_active=True
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print(f"✅ Admin user created:")
        print(f"   Username: {username}")
        print(f"   Email: {email}")
        print(f"   Password: {password}")
        print(f"   Credits: {admin_user.credits}")
        print(f"   User ID: {admin_user.id}")
        
        return admin_user
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def main():
    print("🗄️  Initializing SpatX Database...")
    
    # Initialize database schema
    init_db()
    print("✅ Database schema created")
    
    # Create admin user
    print("\n👤 Creating admin user...")
    create_admin_user()
    
    print("\n✅ Database initialization complete!")
    print("\n📋 Next steps:")
    print("   1. Start backend: bash deploy/start_backend.sh")
    print("   2. Start frontend: bash deploy/start_frontend.sh")
    print("   3. Login with admin/admin123 and create user accounts")

if __name__ == "__main__":
    main()


