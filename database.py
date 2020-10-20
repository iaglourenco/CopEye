# Database operations
import sqlite3
import io
import numpy as np



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
    
    def __init__(self,db_file):
        self.conn = None
        self.conn = self.create_connection(db_file)
        

    # Convert np.ndarray to text and vice-versa when insert and select from SQLite 
    def __adapt_array(self,arr):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())
    
    def __convert_array(self,text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    ################################################################
    
    def create_connection(self,db_file):
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
    
    def run_query(self,query):
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
        self.run_query(self.__query_create_table_fugitivos)
        self.run_query(self.__query_create_table_artigos)
        self.run_query(self.__query_create_table_imagens)
        self.run_query(self.__query_create_table_crimes)
    
    # Inserts
    def insert_fugitivo(self,nome,idade,nivel_perigo):
        if type(nome) is not str or type(idade) is not int or type(nivel_perigo) is not str:
            raise AttributeError()
        sql = """ INSERT INTO fugitivos(nome,idade,nivel_perigo) VALUES (?,?,?) """ 
        cur = self.conn.cursor()
        cur.execute(sql, (nome,idade,nivel_perigo))
        self.conn.commit()
    
    def insert_image(self,id,uri,encoding):
        if type(id) is not int or type(uri) is not str or type(encoding) is not np.ndarray:
            raise AttributeError()
        sql = """ INSERT INTO images(id,uri,encoding) VALUES (?,?,?) """ 
        cur = self.conn.cursor()
        cur.execute(sql, (id,uri,encoding))
        self.conn.commit()
    
        return cur.lastrowid
    
    def insert_crime(self,id,artigo):
        if type(id) is not int or type(artigo) is not int:
            raise AttributeError()
        sql = """ INSERT INTO crimes(id,artigo) VALUES (?,?) """ 
        cur = self.conn.cursor()
        cur.execute(sql, (id,artigo))
        self.conn.commit()
    
    def insert_artigo(self,artigo,descricao,pena=""):
    
        if type(artigo) is not int or type(descricao) is not str or type(pena) is not str:
            raise AttributeError()
        sql = """ INSERT INTO artigos(artigo,descricao,pena) VALUES (?,?,?) """ 
        cur = self.conn.cursor()
        cur.execute(sql, (artigo,descricao,pena))
        self.conn.commit()
    
    #Selects
    def select_all(self,table_name):
    
        try:
            cur = self.conn.cursor()
            cur.execute('SELECT * FROM {}'.format(table_name))
            return cur.fetchall()
    
        except sqlite3.Error as e:
            print(e)
        return None
    
    def select_where(self,table_name,where_clause):
        try:
            cur = self.conn.cursor()
            cur.execute('SELECT * FROM {} WHERE {};'.format(table_name,where_clause))
            return cur.fetchall()
        except sqlite3.Error as e:
            print(e)
        return None
