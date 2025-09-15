from faker import Faker
from dotenv import load_dotenv
import os
from sqlalchemy.orm import Session
from sqlalchemy import func, select
from sqlalchemy_utils import database_exists
import streamlit as st
from datetime import date

from mefa.dashboard.service import db


def create_address(session: Session):
    fake = Faker()
    for i in range(1000):
        address = db.Address(
            street=fake.street_name(),
            city=fake.city(),
            country=fake.country(),
        )
        session.add(address)

    session.commit()


def create_people(session: Session):
    fake = Faker()
    for i in range(1000):
        stmt = select(db.Address.id).order_by(func.random()).limit(1)
        address_id = session.execute(stmt).scalar_one()

        person = db.Person(
            name=fake.name(),
            age=int(fake.numerify("##")),
            annual_income=float(fake.numerify("###.##")),
            address_id=address_id,
        )
        session.add(person)

    session.commit()


@st.cache_resource
def restart_db():
    print("Creating sqlite database and generating fake data")
    db_path = "sqlite:///data.db"
    if not database_exists(db_path):
        engine = db.new_engine(db_path)
        db.Base.metadata.drop_all(engine)
        db.Base.metadata.create_all(engine)

        with Session(engine) as s:
            create_address(s)
            create_people(s)

    today = date.today()
    return today


if __name__ == "__main__":
    load_dotenv(".env")
    db_path = os.environ["ST_DB_PATH"]
    engine = db.new_engine(db_path)
    db.Base.metadata.drop_all(engine)
    db.Base.metadata.create_all(engine)

    with Session(engine) as s:
        create_address(s)
        create_people(s)
