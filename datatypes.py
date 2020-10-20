from numpy import ndarray
class Fugitivo:
    def __init__(self,nome: str,idade: int,nivel_perigo: str):
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
    def __init__(self,id: str,uri:str,encoding: ndarray):
        self.id = id
        self.uri = uri
        self.encoding = encoding
