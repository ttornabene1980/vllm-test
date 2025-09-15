import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

from sqlalchemy import Engine, ForeignKey, create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy_utils import create_database, database_exists


class Base(DeclarativeBase):
    pass


class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    street: Mapped[str]
    city: Mapped[str]
    country: Mapped[str]

    people: Mapped[list["Person"]] = relationship(back_populates="address")

    def __str__(self) -> str:
        return f"{self.street}, {self.city}"


class Person(Base):
    __tablename__ = "person"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    age: Mapped[int]
    annual_income: Mapped[float]
    likes_soccer: Mapped[bool] = mapped_column(default=False)
    address_id: Mapped[int] = mapped_column(ForeignKey("address.id"))

    address: Mapped[Address] = relationship(back_populates="people")

    def __str__(self):
        return self.name


def db_create():
    db: SQLDatabase = SQLDatabase.from_uri(
        "postgresql+psycopg2://sorgente:sorgente@192.168.1.98:5432/sorgente"
    )
    return db
    # tables = db.get_table_names()
    # print("db :", tables)

def new_engine(db_path: str):
    engine = create_engine(db_path)
    return engine


def create_tables(engine: Engine):
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    db_path = "sqlite:///data.db"
    if not database_exists(db_path):
        create_database(db_path)
        url = make_url(db_path)
        print(
            f"Database created on {url.host}, port {url.port}, database {url.database}"
        )

    engine = new_engine(db_path)
    create_tables(engine)
    print("criado")
