score = [85,45,70,55,95]

results = list(map(lambda x: '합격' if x >=60 else '불합격', score)) 
#map 함수로 score에 85,45,70,55,95 값을 집어놓고 x 에 대입  x 조건문 실행 

print(results)

# ['합격', '불합격', '합격', '불합격', '합격']33