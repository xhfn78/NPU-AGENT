def multiply(x):
    return x*10

numbers = [1,2,3,4]

result = map(multiply,numbers)
print(result)
#<map object at 0x0000015D619A6230> #이터레이터 형식의 값으로 반환됨

print(list(result)) #[10, 20, 30, 40] list에 넣어서 값을 출력해야 리스트로 반환값이나옴