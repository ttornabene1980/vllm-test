import streamlit as st
from streamlit_sql import show_sql_ui
from dotenv import load_dotenv
import os
import sys
from datetime import date
from sqlalchemy import select
import pandas as pd
from mefa.dashboard.service import db, restart_db

sys.path.append(os.path.abspath("."))

st.set_page_config("Example streamlit_sql app", layout="wide")
load_dotenv(".env")

restarted_date = restart_db.restart_db()
today = date.today()
if today > restarted_date:
    restart_db.restart_db.clear()  # pyright: ignore
    restart_db.restart_db()

db_path = "sqlite:///data.db"
conn = st.connection("sql", url=db_path)

header_col1, header_col2 = st.columns([8, 4])
header_col1.header("Example application using streamlit_sql")
 
def Person():
    key_prefix="Person"
    def fill_by_age(row: pd.Series):
        if row.age > 30:
            style = "background-color: cyan"
        else:
            style = "background-color: pink"
        result = [style] * len(row)
        return result
    stmt = (
        select(
            db.Person.id,
            db.Person.name,
            db.Person.age,
            db.Person.annual_income,
            db.Person.likes_soccer,
            db.Address.city,
            db.Address.country,
        )
        .select_from(db.Person)
        .join(db.Address)
        .where(db.Person.age > 10)
        .order_by(db.Person.name)
    )
    show_sql_ui(
        conn=conn,
        read_instance=stmt,
        edit_create_model=db.Person,
        rolling_total_column="annual_income",
        available_filter=["name", "country"],
        style_fn=fill_by_age,
         base_key=key_prefix
    )

def Address():
    key_prefix="Address"
    stmt = select(db.Address)
    show_sql_ui(
        conn=conn,
        read_instance=stmt,
        edit_create_model=db.Address,
        update_show_many=True,
        base_key=key_prefix
    )

tab1, tab2 = st.tabs(["Person Table", "Address Table"])

with tab1:
    with st.container():
        Person()
        
with tab2:
    with st.container():
        Address( )

    # # Tabs interne / sub-menu
    # tabs = ["Tab 1", "Tab 2", "Tab 3"]
    # current_tab = st.query_params.get("tab", ["Tab 1"])[0]
    # selected_tab = st.radio("Seleziona Tab", tabs, index=tabs.index(current_tab), horizontal=True)
    # # Aggiorna URL anche con il tab
    # if selected_tab != current_tab:
    #     st.experimental_set_query_params(  tab=selected_tab)
    # # Contenuto tabs
    # if selected_tab == "Tab 1":
    #     st.write("Contenuto Tab 1")
        
# 1️⃣ Upload Datasets
# Users upload one or two CSV files, which are converted into SQL tables.
# 2️⃣ Enter a Natural Language Query
# Example:
# “Find all employees earning more than $50,000 per year.”
# 3️⃣ DeepSeek Generates SQL
# The query is structured using LangChain
# DeepSeek analyzes the dataset
# A MySQL query is generated
# 4️⃣ SQL Query is Displayed
# Users can copy and execute the SQL in their database.

# pages = [
#     st.Page(Person, title="Person Table"),
#     st.Page(Address, title="Address Table"),
# ]
# page = st.navigation(pages)
# page.run()