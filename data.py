import sqlite3
from sqlite3 import Error
import pandas as pd


def create_connection(path):
    con = None
    try:
        con = sqlite3.connect(path)
        print(f"Соединение с SQLite DB '{path}' успешно !")
    except Error as e:
        print(f"Случилась ошибка '{e}'")
    finally:
        return con


def db(con, table):
    data = pd.read_sql(f"SELECT * FROM '{table}';", con, index_col='param', coerce_float=True)
    print(f"Таблица '{table}' загружена:")
    return data
