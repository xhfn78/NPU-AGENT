from typing import Dict,TypedDict


sample_dict : Dict[str,str] = {
    'name' : '김말똥',
    'age' : '39',
    'job' : '회사원'
}

print(sample_dict)
#{'name': '김말똥', 'age': '39', 'job': '회사원'}
##############################################

class Person(TypedDict):
    name : str
    age : int
    job : str   

person = Person(name='홍길동',age=40,job='도적')

print(person)
# {'name': '홍길동', 'age': 40, 'job': '도적'}


person2 : Person = { 'name': '김개똥','age':20,'job':'전사'}#Person은 타입힌트 
print(person2)
