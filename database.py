# Database operations
import sqlite3
import io
import numpy as np


class Fugitivo:
    def __init__(self,nome: str,idade: int,nivel_perigo: str,id=-1):
        self.nome = nome
        self.idade = idade
        self.nivel_perigo = nivel_perigo

class Crime:    
    def __init__(self,id: str,artigo: int):
        self.id = id
        self.artigo = artigo

class Artigo: 
    def __init__(self,artigo: int,descricao: str,pena: str):
        self.artigo = artigo
        self.descricao = descricao
        self.pena = pena

class Shot:
    def __init__(self,id: int,uri:str,encoding: np.ndarray):
        self.id = id
        self.uri = uri
        self.encoding = encoding

class CopEyeDatabase:

    __query_create_table_fugitivos = """CREATE TABLE IF NOT EXISTS fugitivos(
                                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL ,
                                nome VARCHAR NOT NULL, 
                                idade INTEGER NOT NULL,
                                nivel_perigo VARCHAR NOT NULL
                                );"""
    
    __query_create_table_imagens = """CREATE TABLE IF NOT EXISTS imagens (
                                id INTEGER NOT NULL,
                                uri VARCHAR NOT NULL,
                                encoding array NOT NULL,
                                FOREIGN KEY(id) REFERENCES fugitivos(id)
                                );"""
    
    __query_create_table_crimes = """CREATE TABLE IF NOT EXISTS crimes (
                                id INTEGER NOT NULL,
                                artigo INTEGER NOT NULL,
                                FOREIGN KEY(id) REFERENCES fugitivos(id),
                                FOREIGN KEY(artigo) REFERENCES artigos(artigo)
                                );"""
    
    __query_create_table_artigos = """CREATE TABLE IF NOT EXISTS artigos (
                                artigo INTEGER PRIMARY KEY NOT NULL ,
                                descricao VARCHAR,
                                pena VARCHAR
                                );"""
    
    def __init__(self,db_file: str):
        self.conn = None
        self.conn = self.create_connection(db_file)
    



    def __adapt_array(self,arr):
        """
        Convert np.ndarray to text and vice-versa when insert and select from SQLite \n
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    
    def __convert_array(self,text):
        """
        Convert np.ndarray to text and vice-versa when insert and select from SQLite \n
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    ################################################################
    
    def create_connection(self,db_file: str):
        """ Create a database connection to a SQLite database """
        try:
            sqlite3.register_adapter(np.ndarray,self.__adapt_array)
            sqlite3.register_converter("array",self.__convert_array)
            
            if self.conn is not None:
                self.conn.close()

            return sqlite3.connect(db_file,detect_types=sqlite3.PARSE_DECLTYPES)
    
        except sqlite3.Error as e:
            print(e)
        return None									

    def get_connection(self):
        """Returns the instance connection to the database"""
        return self.conn    
    
    def run_query(self,query:str):
        """Runs the query on database and returns the result"""
        if self.conn is not None:
            try:
                c = self.conn.cursor()
                c.execute(query)
                return c.fetchall()
            except sqlite3.Error as e:
                print(e)
    
        else:
            print("Can't create database connection")
    
    def init_database(self):
        """Create all tables of DefaultDB"""
        self.run_query(self.__query_create_table_fugitivos)
        self.run_query(self.__query_create_table_artigos)
        self.run_query(self.__query_create_table_imagens)
        self.run_query(self.__query_create_table_crimes)
    
    # Inserts
    def insert_fugitivo(self,fugitivo: Fugitivo):
        """Insert a new fugitivo into fugitives table"""
        if type(fugitivo) is not Fugitivo:
            raise TypeError()

        sql = """ INSERT INTO fugitivos(nome,idade,nivel_perigo) VALUES (?,?,?) """ 
        cur = self.conn.cursor()    
        cur.execute(sql, (fugitivo.nome,fugitivo.idade,fugitivo.nivel_perigo))
        self.conn.commit()
        return cur.lastrowid
    
    def insert_image(self,shot: Shot):
        """Insert a new shot into imagens table"""

        if type(shot) is not Shot:
            raise AttributeError()
        sql = """ INSERT INTO imagens(id,uri,encoding) VALUES (?,?,?) """ 
        cur = self.conn.cursor()
        cur.execute(sql, (shot.id,shot.uri,shot.encoding))
        self.conn.commit()
    
        return cur.lastrowid
    
    def insert_crime(self,crime:Crime):
        """Insert a new crime into crimes table"""

        if type(crime) is not Crime:
            raise AttributeError()
        sql = """ INSERT INTO crimes(id,artigo) VALUES (?,?) """ 
        cur = self.conn.cursor()
        cur.execute(sql, (crime.id,crime.artigo))
        self.conn.commit()
    
    def insert_artigo(self,artigo: int,descricao: str,pena=""):
        """Insert a new artigo into artigos table"""
        
        if type(artigo) is not int or type(descricao) is not str or type(pena) is not str:
            raise AttributeError()
        sql = """ INSERT INTO artigos(artigo,descricao,pena) VALUES (?,?,?) """ 
        cur = self.conn.cursor()
        cur.execute(sql, (artigo,descricao,pena))
        self.conn.commit()
    
    #Selects
    def select_all(self,table_name:str):
        """Selects all rows from table with given name"""
        try:
            cur = self.conn.cursor()
            cur.execute('SELECT * FROM {}'.format(table_name))
            return cur.fetchall()
    
        except sqlite3.Error as e:
            print(e)
        return None
    
    def select(self,what_clauses:str,table_names: str,where_clauses: str):
        """Construct a SELECT query and return the result """
        try:
            cur = self.conn.cursor()
            cur.execute('SELECT {} FROM {} WHERE {};'.format(what_clauses,table_names,where_clauses))
            return cur.fetchall()
        except sqlite3.Error as e:
            print(e)
        return None



