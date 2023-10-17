import random
from matplotlib import pyplot as plt

population = 10
answer = [4, 1, 3, 2, 4, 4, 1, 3, 2, 3]
sum = 0
mut_per = 0.6                   #mutation 일어날 확률 : 60%
fit_average = 0
num_break = 0

def b_sort(fit, sol):                               #fitness값을 기준으로 population을 같이 정렬시킴
    end = len(fit) - 1
    while end > 0:
        last_swap = 0
        for i in range(end):
            if fit[i] > fit[i+1]:
                fit[i], fit[i+1] = fit[i+1], fit[i]
                sol[i], sol[i+1] = sol[i+1], sol[i]
                last_swap = i
        end = last_swap

def fit_val(t_sol):           #각 solution의 fitness값 추출
    a = 0
    for i in range(10):
        if t_sol[i] == answer[i] :
            a += 10
    return a

def mutation(sol, num) :                 #mutation 전략 : 각 cell마다 0.5%의 확률로 랜덤한 숫자로 변함
    for i in range(10):
        if random.random() < mut_per:
            sol[num][i] = random.randint(1,4)

def avg_fit(a):              #각 세대에서의 평균 fitness값 계산 및 출력
    avg = a / 10
    return int(avg)

sol = [[random.randint(1,4) for i in range(10)] for j in range(10)]      #초기 랜덤 솔루션 10개 생성
ofs = [[0 for i in range(10)] for j in range(10)]                        #임시자식 soultion 10개 생성

sum = 0
for i in range(10):
    sum += fit_val(sol[i])

f_avg = avg_fit(sum)

ls_gen1_2 = [1]
ls_avgfit1_2 = [f_avg]

for generation in range(2, 1001) :      #세대 반복 시작
    for selection in range(10) :        #자식 soultion 만들기 시작
        list = []
        ran_num = random.randint(0,9)
        for i in range(3) :             #부모에서 무작위 3개 선택
            while ran_num in list:
                ran_num = random.randint(0,9)
            list.append(ran_num)

        a = fit_val(sol[list[0]])       #무작위로 선택된 3개 중 삭제할 것을 fitness proportionate selection방식으로 선택
        b = fit_val(sol[list[1]])
        c = fit_val(sol[list[2]])
        sum = a + b + c
        d_num = random.choices(range(0,3), weights=[sum-a, sum-b, sum-c])
        del list[d_num[0]]

        point = random.randint(1,9)        #crossover point 랜덤으로 선택

        ofs[selection] = sol[list[0]][:point] + sol[list[1]][point:]        #crossover

    for i in range(10):
        mutation(ofs, i)

    #population 뽑기
    temp_pop = 0
    temp_sol = [[0 for i in range(10)] for j in range(20)]      #임시로 전체 population 만듦
    for i in range(10):
        temp_sol[i] = sol[i]
    for j in range(10,20) :
        temp_sol[j] = ofs[j-10]

    temp_fit = [fit_val(temp_sol[i]) for i in range(20)]        #임시 전체 population의 fitness값 배열

    b_sort(temp_fit,temp_sol)                   #fitness값 큰 순서대로 임시 fitness값 배열을 기준으로 임시 전체 population 정렬

    for i in range(10):                     #fitness값이 큰 10개만 살아남아 다음 generation으로 전달
        sol[i] = temp_sol[i+10]

    sum = 0
    for i in range(10):
        sum += fit_val(sol[i])
    fit_average = avg_fit(sum)

    ls_gen1_2.append(generation)
    ls_avgfit1_2.append(fit_average)

    for i in range(10):
        if fit_val(sol[i]) == 100:
            num_break = 1
            break
    if num_break == 1:
        break
for i in range(10):
    print(sol[i], fit_val(sol[i]))
print(generation)
print(fit_average)

plt.plot(ls_gen1_2, ls_avgfit1_2)
plt.xlabel('Generation')
plt.ylabel('Average of fitness value')
plt.show()

